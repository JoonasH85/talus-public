//! GLM-5.3-Flash stdio engine protocol backend (`talus-cli --model GLM serve --engine-protocol`), driven by
//! `serve/server.py --engine strata` or `talus-server`.
//!
//! Prompt cache (talus-livecold exact policy): the recurrent state (KDA + conv) is snapshotted only on the prompt
//! chunk grid (multiples of `GRID` tokens, written by prefill) and never restored from decode-written state; the
//! position-indexed caches (MLA latent, indexer keys / pooled keys, MTP block) are reused up to the snapshot.

use std::io::Write;
use std::time::Instant;

use talus_model::glm::cli::{self as gcli, GlmCli};
use talus_model::glm::engine::{Buf, GlmEngine};
use talus_model::glm::sample::SamplerParams;
use talus_model::glm::spec::{self, GenParams, GenStats, SeqPos, SnapEvent};
use talus_sched::{
    write_done_line, write_pp_line, write_resume_line, write_token_line, CancelToken, DoneStats,
    EngineProtocolBackend, GenCommand,
};

/// Snapshot grid (tokens): a multiple of the prefill window and the k-pool size.
pub const GRID: usize = 512;

/// Snapshot grid at run time: GLM_SNAP_GRID (a multiple of GRID), default GRID.
pub(crate) fn snap_grid() -> usize {
    static G: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *G.get_or_init(|| {
        std::env::var("GLM_SNAP_GRID")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&g| g >= GRID && g % GRID == 0)
            .unwrap_or(GRID)
    })
}

struct Snap {
    pos: usize,
    hash: u64,
    h_prev: Vec<f32>,
    buf: Buf,
    used: u64,
}

pub struct GlmBackend {
    eng: GlmEngine,
    cfg: GlmCli,
    /// tokens whose state is committed in the position-indexed caches (positions 0..hist.len())
    hist: Vec<u32>,
    snaps: Vec<Snap>,
    max_snaps: usize,
    clock: u64,
    log: bool,
    /// TALUS_MEM_SHED_EVERY: 1 = every class, or a comma list of graphs,snapshots,pcache,arenas
    shed_every: Option<String>,
}

fn hash_prefix(t: &[u32]) -> u64 {
    // FNV-1a over the token ids
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &x in t {
        for b in x.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

/// glm-vision fork: the image rows of a request, from the file named by GEN `emb=` (SVE1 records written by the
/// vision encoder, one per image in prompt order: int32 {0x31455653, n_tokens, nx, ny, n_embd} then
/// float32 [n_tokens][n_embd]) placed at the positions GEN `img_pos=` names. Returns the flat rows and the
/// position -> row map.
fn load_images(
    path: &str,
    starts: &[usize],
    n_embd: usize,
    prompt_len: usize,
) -> Result<(Vec<f32>, std::collections::HashMap<usize, usize>, Vec<usize>), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("image rows {path}: {e}"))?;
    let mut rows: Vec<f32> = Vec::new();
    let mut map = std::collections::HashMap::new();
    let mut counts = Vec::new();
    let mut off = 0usize;
    let i32_at = |o: usize| -> Result<i32, String> {
        bytes
            .get(o..o + 4)
            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .ok_or_else(|| "image rows: truncated header".to_string())
    };
    for (k, &p0) in starts.iter().enumerate() {
        if i32_at(off)? as u32 != 0x3145_5653 {
            return Err(format!("image rows: record {k} is not SVE1"));
        }
        let n = i32_at(off + 4)?.max(0) as usize;
        let e = i32_at(off + 16)?.max(0) as usize;
        if e != n_embd {
            return Err(format!("image rows: n_embd {e}, the model has {n_embd}"));
        }
        let body = off + 20;
        let end = body + n * e * 4;
        if end > bytes.len() {
            return Err(format!("image rows: record {k} truncated"));
        }
        if p0 + n > prompt_len {
            return Err(format!("image {k}: rows {p0}..{} past the prompt ({prompt_len})", p0 + n));
        }
        for j in 0..n {
            let r = rows.len() / e;
            if map.insert(p0 + j, r).is_some() {
                return Err(format!("image {k}: position {} used twice", p0 + j));
            }
            let src = &bytes[body + j * e * 4..body + (j + 1) * e * 4];
            rows.extend(src.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])));
        }
        counts.push(n);
        off = end;
    }
    if off != bytes.len() {
        return Err(format!(
            "image rows: {} images named, the file holds more ({} of {} bytes read)",
            starts.len(),
            off,
            bytes.len()
        ));
    }
    Ok((rows, map, counts))
}

/// glm-vision fork: the ids the prompt cache compares - the prompt's, with every image position replaced by a key
/// of its image (content hash) and row, outside the vocabulary (bit 31 set), so a cached prefix only matches the
/// same pictures at the same places.
fn cache_keys(prompt: &[u32], starts: &[usize], counts: &[usize], img_keys: &[u64]) -> Vec<u32> {
    let mut keys = prompt.to_vec();
    for (k, (&p0, &n)) in starts.iter().zip(counts).enumerate() {
        let ik = img_keys.get(k).copied().unwrap_or(0x9e37_79b9_7f4a_7c15 ^ k as u64);
        for j in 0..n {
            let mut h = ik ^ (j as u64).wrapping_mul(0x0100_0000_01b3);
            h ^= h >> 29;
            h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
            h ^= h >> 32;
            if let Some(slot) = keys.get_mut(p0 + j) {
                *slot = 0x8000_0000 | (h as u32 & 0x7fff_ffff);
            }
        }
    }
    keys
}

impl GlmBackend {
    pub fn new(eng: GlmEngine, cfg: GlmCli) -> Self {
        let max_snaps = std::env::var("GLM_PROMPT_SNAPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1);
        Self {
            eng,
            cfg,
            hist: vec![],
            snaps: vec![],
            max_snaps,
            clock: 0,
            log: std::env::var("GLM_SERVE_LOG").is_ok(),
            shed_every: std::env::var("TALUS_MEM_SHED_EVERY")
                .ok()
                .filter(|v| !v.is_empty() && v != "0"),
        }
    }

    fn run(
        &mut self,
        cmd: &GenCommand,
        stop: &CancelToken,
        out: &mut dyn Write,
    ) -> talus_core_api::Result<()> {
        let t0 = Instant::now();
        let prompt: Vec<u32> = cmd.prompt_ids.iter().map(|&x| x as u32).collect();
        if prompt.is_empty() {
            return Err(talus_core_api::CoreError::InvalidArgument {
                reason: "empty prompt".into(),
            });
        }
        // glm-vision fork: image rows for this request (none: a text-only request, exactly as before)
        let keys: Vec<u32> = match cmd.emb.as_deref() {
            Some(path) => {
                let (rows, map, counts) =
                    load_images(path, &cmd.img_pos, self.eng.w.hp.n_embd, prompt.len()).map_err(|e| {
                        talus_core_api::CoreError::InvalidArgument { reason: e }
                    })?;
                if self.log {
                    eprintln!("glm-serve: {} image(s), {} rows {:?}", counts.len(), map.len(), counts);
                }
                self.eng.set_images(rows, map)?;
                cache_keys(&prompt, &cmd.img_pos, &counts, &cmd.img_keys)
            }
            None => prompt.clone(),
        };
        let r = self.run_keyed(cmd, stop, out, t0, prompt, keys);
        self.eng.clear_images();
        r
    }

    fn run_keyed(
        &mut self,
        cmd: &GenCommand,
        stop: &CancelToken,
        out: &mut dyn Write,
        t0: Instant,
        prompt: Vec<u32>,
        keys: Vec<u32>,
    ) -> talus_core_api::Result<()> {
        // common prefix with the committed history (at least one prompt token is evaluated for its logits)
        let mut l = 0;
        if cmd.use_cache {
            while l < keys.len() && l < self.hist.len() && keys[l] == self.hist[l] {
                l += 1;
            }
        }
        let l = l.min(prompt.len() - 1);
        // the newest snapshot at or below l whose prefix matches
        let mut best: Option<usize> = None;
        for (i, s) in self.snaps.iter().enumerate() {
            if s.pos <= l
                && s.pos > best.map(|b| self.snaps[b].pos).unwrap_or(0)
                && s.hash == hash_prefix(&keys[..s.pos])
            {
                best = Some(i);
            }
        }
        let mut seq = SeqPos::default();
        match best {
            Some(i) => {
                let buf = self.snaps[i].buf;
                self.eng.snap_restore(&buf)?;
                self.clock += 1;
                self.snaps[i].used = self.clock;
                seq.n_past = self.snaps[i].pos;
                seq.h_prev = self.snaps[i].h_prev.clone();
            }
            None => {
                self.eng.reset()?;
            }
        }
        let reuse = seq.n_past;
        write_resume_line(out, reuse).ok();
        // grid snapshots of this prompt: the last grid point before its end (and the one before it with 2+ slots)
        let end = prompt.len();
        let grid = snap_grid();
        let last_grid = (end.saturating_sub(1) / grid) * grid;
        let points: Vec<usize> = [last_grid, last_grid.saturating_sub(grid)]
            .into_iter()
            .take(self.max_snaps.max(1))
            .filter(|&g| g > reuse && g > 0)
            .collect();
        let mtp = self.cfg.mtp;
        let first = {
            // the snapshot copy must happen at the point itself (the state moves on with the next window)
            let snaps = &mut self.snaps;
            let max_snaps = self.max_snaps;
            let clock = &mut self.clock;
            let prompt_ref = &keys; // glm-vision fork: snapshots are named by the cache keys
            let mut pending: Option<usize> = None; // slot being filled
            let mut cb2 = |eng: &mut GlmEngine,
                           ev: SnapEvent|
             -> talus_core_api::Result<Option<Buf>> {
                if max_snaps == 0 {
                    return Ok(None);
                }
                match ev {
                    SnapEvent::Want(pos) => {
                        let hash = hash_prefix(&prompt_ref[..pos]);
                        *clock += 1;
                        if let Some(s) = snaps.iter_mut().find(|s| s.pos == pos && s.hash == hash) {
                            s.used = *clock;
                            return Ok(None); // already held
                        }
                        let idx = if snaps.len() < max_snaps {
                            let buf = eng.snap_alloc()?;
                            snaps.push(Snap {
                                pos: 0,
                                hash: 0,
                                h_prev: vec![],
                                buf,
                                used: 0,
                            });
                            snaps.len() - 1
                        } else {
                            (0..snaps.len()).min_by_key(|&i| snaps[i].used).unwrap()
                        };
                        // invalid until Done (an error in between leaves it unusable)
                        snaps[idx].pos = usize::MAX;
                        pending = Some(idx);
                        Ok(Some(snaps[idx].buf))
                    }
                    SnapEvent::Done(pos, h) => {
                        if let Some(idx) = pending.take() {
                            let s = &mut snaps[idx];
                            s.pos = pos;
                            s.hash = hash_prefix(&prompt_ref[..pos]);
                            s.h_prev = h.to_vec();
                            s.used = *clock;
                        }
                        Ok(None)
                    }
                }
            };
            spec::prefill_with(
                &mut self.eng,
                &mut seq,
                &prompt[reuse..],
                mtp,
                &points,
                &mut cb2,
            )?
        };
        if std::env::var("GLM_LOGITS_HASH").as_deref() == Ok("1") {
            // diagnostics (rd-mempr warm vs cold): FNV-1a over the bits of the last prompt token's logits
            let mut h: u64 = 0xcbf2_9ce4_8422_2325;
            for v in first.iter() {
                for b in v.to_bits().to_le_bytes() {
                    h ^= b as u64;
                    h = h.wrapping_mul(0x0100_0000_01b3);
                }
            }
            eprintln!(
                "glm-serve: logits-hash {h:016x} prompt {} reuse {reuse}",
                prompt.len()
            );
        }
        let prompt_ms = t0.elapsed().as_secs_f64() * 1e3;
        let fresh = (prompt.len() - reuse) as i64;
        write_pp_line(out, prompt.len() as i64, fresh, prompt_ms).ok();

        let samp = SamplerParams {
            temp: cmd.temperature,
            top_k: cmd.top_k,
            top_p: cmd.top_p,
            min_p: cmd.min_p,
            seed: cmd.seed,
        };
        let p = GenParams {
            max_new: cmd.max_new.max(1) as usize,
            samp,
            mtp,
            eos: self.eng.w.hp.eos.clone(),
            ignore_eos: cmd.ignore_eos,
            ctx: prompt.clone(),
        };
        let mut st = GenStats::default();
        let mut cancelled = false;
        let n0 = seq.n_past;
        let gen = spec::generate(&mut self.eng, &mut seq, &first, &p, &mut st, |t| {
            write_token_line(out, t as i32).ok();
            if stop.is_cancelled() {
                cancelled = true;
                return false;
            }
            true
        })?;
        // committed history: the prompt + the generated tokens the trunk has processed
        let processed = seq.n_past - n0;
        self.hist = keys.clone();
        self.hist
            .extend_from_slice(&gen[..processed.min(gen.len())]);
        let finish: &'static str = if cancelled {
            "cancel"
        } else if st.finish == "stop" {
            "stop"
        } else {
            "length"
        };
        if self.log {
            eprintln!(
                "glm-serve: prompt {} (reuse {reuse}) {:.0} ms | gen {} {:.0} ms ({:.1} tok/s) accept {}/{} | avail {:.2} GiB{}",
                prompt.len(),
                prompt_ms,
                gen.len(),
                st.decode_ms,
                gen.len() as f64 / st.decode_ms.max(1e-3) * 1e3,
                st.accepted,
                st.drafted,
                talus_model::glm::weights::mem_available_bytes() as f64 / (1u64 << 30) as f64,
                talus_mem::pressure::global()
                    .map(|p| format!(" | pressure: {}", p.summary()))
                    .unwrap_or_default()
            );
        }
        write_done_line(
            out,
            &DoneStats {
                produced: gen.len() as i64,
                prompt_tokens: prompt.len(),
                prompt_ms,
                decode_ms: st.decode_ms,
                finish,
                accepted: st.accepted as i64,
                offered: st.drafted as i64,
                reuse,
                fresh,
            },
        )
        .ok();
        Ok(())
    }
}

/// #40 memory pressure: the backend's rebuildable memory is the engine's (graphs, prefill arenas) plus the prompt
/// snapshots. Dropping a snapshot costs the next turn a re-read of its prompt, and (warm != cold on GLM) may change its
/// text, so snapshots are an inexact class, shed last.
impl talus_mem::pressure::Shed for GlmBackend {
    fn shed(&mut self, class: talus_mem::pressure::ShedClass) -> u64 {
        use talus_mem::pressure::ShedClass;
        match class {
            ShedClass::PromptSnapshots => {
                let mut n = 0u64;
                for s in std::mem::take(&mut self.snaps) {
                    if self.eng.snap_free(&s.buf).is_ok() {
                        n += s.buf.bytes as u64;
                    }
                }
                n
            }
            _ => self.eng.shed(class),
        }
    }

    /// GLM's warm resume is not bitwise the cold read (rd-mempr wc, 2026-10-31): a resumed window below 1152 rows
    /// (288 experts x pf_w16_moe_min_rows 32) runs the MMQ expert path where the cold run's 4096-row window runs the
    /// fp16 (w16) experts. So a dropped snapshot can change the next turn's text (it completes as a cold run); shed
    /// last (operator option A). Exact when every window takes w16 (STRATA_PF_W16_MOE_MIN=0) or the snapshot grid is
    /// the chunk (GLM_SNAP_GRID=4096).
    fn inexact(&self, class: talus_mem::pressure::ShedClass) -> bool {
        class == talus_mem::pressure::ShedClass::PromptSnapshots
            && std::env::var("STRATA_PF_W16_MOE_MIN").as_deref() != Ok("0")
            && super::glm_serve::snap_grid() % talus_model::glm::engine::pch() != 0
    }
}

impl GlmBackend {
    /// TALUS_MEM_SHED_EVERY=1 (test knob): after every request shed every class and log what each released, so a
    /// run can show that shedding changes no output (and measure the classes).
    fn shed_probe(&mut self) {
        use talus_mem::pressure::{Shed, SHED_ORDER};
        let mut parts = vec![];
        let want = self.shed_every.clone().unwrap_or_default();
        for c in SHED_ORDER {
            let key = match c {
                talus_mem::pressure::ShedClass::DecodeGraphs => "graphs",
                talus_mem::pressure::ShedClass::PromptSnapshots => "snapshots",
                talus_mem::pressure::ShedClass::PromptCacheRam => "pcache",
                talus_mem::pressure::ShedClass::TransientArenas => "arenas",
            };
            if want != "1" && !want.split(',').any(|w| w == key) {
                continue;
            }
            let a = talus_model::glm::weights::mem_available_bytes();
            let n = self.shed(c);
            std::thread::sleep(std::time::Duration::from_millis(50));
            let b = talus_model::glm::weights::mem_available_bytes();
            parts.push(format!(
                "{} {} (MemAvailable {:+.1} MiB)",
                c.name(),
                if n > 0 { "shed" } else { "none" },
                (b as f64 - a as f64) / (1u64 << 20) as f64
            ));
        }
        eprintln!("glm-serve: shed probe: {}", parts.join(", "));
    }
}

impl EngineProtocolBackend for GlmBackend {
    fn max_context(&self) -> i32 {
        self.eng.cfg.ctx as i32
    }
    fn n_vocab(&self) -> i32 {
        self.eng.w.hp.n_vocab as i32
    }
    fn handle_gen(
        &mut self,
        cmd: &GenCommand,
        stop: &CancelToken,
        out: &mut dyn Write,
    ) -> std::io::Result<()> {
        // #40: admission at request start: shed in the fixed order when the host is near the floor; still under it
        // -> this request fails (ERR insufficient_memory), the server keeps serving
        let r = match talus_mem::pressure::admit(self) {
            Ok(()) => self.run(cmd, stop, out),
            Err(e) => Err(e.into()),
        };
        if let Err(e) = r {
            writeln!(out, "{}", talus_mem::pressure::err_line(&e.to_string()))?;
            out.flush()?;
            // state is unknown after an error: drop the history so the next request starts cold
            self.hist.clear();
            if talus_mem::pressure::is_insufficient_memory(&e.to_string()) {
                // what the failed request may have left behind: the snapshot slot it was filling is invalid anyway
                self.snaps.retain(|s| s.pos != usize::MAX);
            }
        }
        if self.shed_every.is_some() {
            self.shed_probe();
        }
        Ok(())
    }
}

/// GLM entry from talus-cli: serve mode here (needs talus-sched), other modes in talus_model::glm::cli.
pub fn run(argv: &[String]) -> i32 {
    let mut c = match gcli::parse(argv) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("talus-glm: {e}");
            return 2;
        }
    };
    if !c.serve {
        return gcli::run(argv);
    }
    talus_model::glm::weights::spawn_watchdog((c.minavail_gib * (1u64 << 30) as f64) as u64);
    let mut eng = match gcli::load_engine(&mut c) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("talus-glm: {e}");
            return 1;
        }
    };
    if let Some(dv) = &c.draft_vocab {
        match gcli::read_draft_vocab(dv).and_then(|ids| eng.set_draft_vocab(&ids)) {
            Ok(()) => {}
            Err(e) => {
                eprintln!("talus-glm: draft vocab: {e}");
                return 1;
            }
        }
    }
    let _ = eng.reset();
    let mut be = GlmBackend::new(eng, c);
    talus_sched::serve_stdio(&mut be)
}
