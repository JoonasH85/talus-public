//! glm-vision fork: the image encoder (a resident `maya-vision` / `strata-vision` subprocess: llama.cpp's mtmd with the
//! model's mmproj) and the image rows the engine reads in place of the image tokens.
//!
//! Based on kva19's vision glue for Flash-Next (github.com/nsbradley88/talus-public PR #7, MIT), with the fixes from
//! the maintainer's review of it: files named by a hash of the whole picture (same-size pictures never share rows),
//! a picture that can't be read is an error (HTTP 400), never an answer without it; the image tokens are put in at
//! token level by the server (render_and_encode_chat), not re-tokenized from text; flash attention is left on (the
//! encoder's default); only base64 pictures (no URL fetching).
//!
//! Encoder protocol (tools/vision/strata_vision.cpp in project-maya / Strata):
//!   encoder -> "READY <n_embd>" once loaded
//!   us      -> "ENC <image path> <output path>"
//!   encoder -> "OK <n_tokens> <nx> <ny> <ms>" | "ERR <message>"
//!   us      -> "QUIT"
//! The output file: int32 {0x31455653 'SVE1', n_tokens, nx, ny, n_embd}, then float32 [n_tokens][n_embd].

use std::collections::{HashMap, VecDeque};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdout, Command, Stdio};
use std::sync::Mutex;

use crate::config::VisionBlock;

/// What the engine needs for a request with pictures (GEN keys `emb=`, `img_pos=`, `img_keys=`).
#[derive(Clone, Debug, PartialEq)]
pub struct ImageRequest {
    /// one file of SVE1 records, one per picture in prompt order
    pub emb: String,
    /// the prompt position of each picture's first row
    pub pos: Vec<usize>,
    /// a content hash per picture (the prompt cache's key for its rows)
    pub keys: Vec<u64>,
}

pub struct Encoded {
    pub path: PathBuf,
    pub n_tokens: usize,
    pub key: u64,
}

struct Proc {
    child: Child,
    out: BufReader<ChildStdout>,
}

impl Drop for Proc {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

pub struct VisionEncoder {
    cfg: VisionBlock,
    dir: PathBuf,
    proc: Mutex<Option<Proc>>,
    /// picture key -> (rows file, rows); bounded, oldest out first
    cache: Mutex<(HashMap<u64, (PathBuf, usize)>, VecDeque<u64>)>,
    /// the last picture the encoder process handled (idle shutdown, `idle_secs`)
    last_use: Mutex<std::time::Instant>,
}

const CACHE_MAX: usize = 64;

/// 64-bit key of a picture's bytes: two FNV-1a variants folded (a cache name, not a security hash).
pub fn content_key(data: &[u8]) -> u64 {
    let (mut a, mut b): (u64, u64) = (0xcbf2_9ce4_8422_2325, 0x8422_2325_cbf2_9ce4);
    for &x in data {
        a = (a ^ x as u64).wrapping_mul(0x0100_0000_01b3);
        b = (b ^ x as u64).wrapping_mul(0x0000_0100_0000_01b3 ^ 0x9e37);
    }
    a ^ b.rotate_left(29) ^ (data.len() as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
}

/// Standard or URL-safe base64, padding and whitespace ignored; None on any other byte.
pub fn base64_decode(s: &str) -> Option<Vec<u8>> {
    fn val(c: u8) -> Option<u32> {
        match c {
            b'A'..=b'Z' => Some((c - b'A') as u32),
            b'a'..=b'z' => Some((c - b'a' + 26) as u32),
            b'0'..=b'9' => Some((c - b'0' + 52) as u32),
            b'+' | b'-' => Some(62),
            b'/' | b'_' => Some(63),
            _ => None,
        }
    }
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let (mut acc, mut bits) = (0u32, 0u32);
    for b in s.bytes() {
        if matches!(b, b'=' | b'\n' | b'\r' | b' ' | b'\t') {
            continue;
        }
        acc = (acc << 6) | val(b)?;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push(((acc >> bits) & 0xff) as u8);
        }
    }
    Some(out)
}

impl VisionEncoder {
    pub fn new(cfg: VisionBlock) -> std::io::Result<Self> {
        let dir = std::env::temp_dir().join(format!("talus-vision-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        Ok(Self {
            cfg,
            dir,
            proc: Mutex::new(None),
            cache: Mutex::new((HashMap::new(), VecDeque::new())),
            last_use: Mutex::new(std::time::Instant::now()),
        })
    }

    /// Quit the encoder process after `idle_secs` without a picture (`idle_secs` 0: a 5 s safety net behind
    /// [`Self::release_after_use`]); never while it encodes: the process lock is only tried. The next picture starts
    /// it again (~1 s).
    fn reap_idle(&self) {
        if self.last_use.lock().unwrap().elapsed().as_secs() < self.cfg.idle_secs.max(5) {
            return;
        }
        self.stop_proc();
    }

    /// `idle_secs` 0: quit the encoder as soon as a request's pictures are encoded, so its memory is back before
    /// the engine reads the request (at 128K context a resident encoder takes the engine below its shed line).
    pub fn release_after_use(&self) {
        if self.cfg.idle_secs == 0 {
            self.stop_proc();
        }
    }

    fn stop_proc(&self) {
        let Ok(mut g) = self.proc.try_lock() else {
            return;
        };
        if let Some(mut p) = g.take() {
            if let Some(stdin) = p.child.stdin.as_mut() {
                let _ = writeln!(stdin, "QUIT");
                let _ = stdin.flush();
            }
            for _ in 0..30 {
                if matches!(p.child.try_wait(), Ok(Some(_))) {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            drop(p); // kills it if QUIT was not enough
            if self.cfg.idle_secs > 0 {
                eprintln!("[vision] encoder stopped after {} s without a picture", self.cfg.idle_secs);
            }
        }
    }

    fn spawn(&self) -> Result<Proc, String> {
        let mut cmd = Command::new(&self.cfg.exe);
        cmd.arg("--mmproj").arg(&self.cfg.mmproj).arg("--model").arg(&self.cfg.model);
        if self.cfg.gpu {
            cmd.arg("--gpu");
        }
        if let Some(t) = self.cfg.threads {
            cmd.arg("--threads").arg(t.to_string());
        }
        if let Some(m) = self.cfg.max_tokens {
            cmd.arg("--max-tokens").arg(m.to_string());
        }
        let log = std::fs::File::create(self.dir.join("encoder.log"))
            .map_err(|e| format!("vision encoder log: {e}"))?;
        // the encoder brings its own ROCm (its RUNPATH): not the engine's library path
        cmd.env_remove("LD_LIBRARY_PATH");
        cmd.current_dir(&self.dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::from(log));
        let mut child = cmd.spawn().map_err(|e| format!("vision encoder did not start: {e}"))?;
        let out = child.stdout.take().ok_or("vision encoder: no stdout")?;
        let mut p = Proc {
            child,
            out: BufReader::new(out),
        };
        let mut line = String::new();
        p.out
            .read_line(&mut line)
            .map_err(|e| format!("vision encoder did not start: {e}"))?;
        if !line.starts_with("READY") {
            return Err(format!(
                "vision encoder did not start: {} (see {})",
                line.trim(),
                self.dir.join("encoder.log").display()
            ));
        }
        eprintln!("[vision] encoder ready: {}", line.trim());
        Ok(p)
    }

    /// Encode one picture (its file bytes) into its rows file.
    pub fn encode(&self, data: &[u8]) -> Result<Encoded, String> {
        if data.is_empty() {
            return Err("the image is empty".into());
        }
        let key = content_key(data);
        if let Some((path, n)) = self.cache.lock().unwrap().0.get(&key) {
            if path.exists() {
                return Ok(Encoded {
                    path: path.clone(),
                    n_tokens: *n,
                    key,
                });
            }
        }
        let img = self.dir.join(format!("{key:016x}.img"));
        let sve = self.dir.join(format!("{key:016x}.sve"));
        std::fs::write(&img, data).map_err(|e| format!("image temp file: {e}"))?;
        *self.last_use.lock().unwrap() = std::time::Instant::now();
        let mut guard = self.proc.lock().unwrap();
        if guard.is_none() {
            *guard = Some(self.spawn()?);
        }
        let reply = {
            let p = guard.as_mut().unwrap();
            let stdin = p.child.stdin.as_mut().ok_or("vision encoder: no stdin")?;
            writeln!(stdin, "ENC {} {}", img.display(), sve.display())
                .and_then(|_| stdin.flush())
                .map_err(|e| format!("vision encoder: {e}"))
                .and_then(|_| {
                    let mut line = String::new();
                    p.out
                        .read_line(&mut line)
                        .map_err(|e| format!("vision encoder: {e}"))
                        .map(|_| line.trim().to_string())
                })
        };
        let _ = std::fs::remove_file(&img);
        let line = match reply {
            Ok(l) if !l.is_empty() => l,
            Ok(_) | Err(_) => {
                *guard = None; // a dead encoder is started again on the next picture
                return Err("the vision encoder stopped".into());
            }
        };
        if let Some(msg) = line.strip_prefix("ERR") {
            return Err(msg.trim().to_string());
        }
        let n: usize = line
            .strip_prefix("OK ")
            .and_then(|r| r.split_whitespace().next())
            .and_then(|x| x.parse().ok())
            .ok_or_else(|| format!("vision encoder: unexpected reply {line:?}"))?;
        if n == 0 {
            return Err("the image gave no tokens".into());
        }
        *self.last_use.lock().unwrap() = std::time::Instant::now();
        drop(guard);
        let mut c = self.cache.lock().unwrap();
        c.0.insert(key, (sve.clone(), n));
        c.1.push_back(key);
        while c.1.len() > CACHE_MAX {
            if let Some(old) = c.1.pop_front() {
                if let Some((p, _)) = c.0.remove(&old) {
                    let _ = std::fs::remove_file(p);
                }
            }
        }
        Ok(Encoded {
            path: sve,
            n_tokens: n,
            key,
        })
    }

    /// One file of the pictures' rows files in prompt order, named by their keys (written once).
    pub fn combine(&self, parts: &[Encoded]) -> Result<PathBuf, String> {
        let mut name = 0xcbf2_9ce4_8422_2325u64;
        for p in parts {
            name = (name ^ p.key).wrapping_mul(0x0100_0000_01b3).rotate_left(17);
        }
        let out = self.dir.join(format!("req-{name:016x}-{}.sve", parts.len()));
        if !out.exists() {
            let tmp = out.with_extension("tmp");
            let mut f = std::fs::File::create(&tmp).map_err(|e| format!("image rows file: {e}"))?;
            for p in parts {
                let b = std::fs::read(&p.path).map_err(|e| format!("image rows file: {e}"))?;
                f.write_all(&b).map_err(|e| format!("image rows file: {e}"))?;
            }
            drop(f);
            std::fs::rename(&tmp, &out).map_err(|e| format!("image rows file: {e}"))?;
        }
        Ok(out)
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }
}

impl Drop for VisionEncoder {
    fn drop(&mut self) {
        if let Some(mut p) = self.proc.lock().unwrap().take() {
            if let Some(stdin) = p.child.stdin.as_mut() {
                let _ = writeln!(stdin, "QUIT");
                let _ = stdin.flush();
            }
        }
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// The encoder from the config's optional "vision" block; None = text only (exactly as without the fork).
pub fn encoder_from_config(cfg: &crate::config::ServerConfig) -> Option<std::sync::Arc<VisionEncoder>> {
    let v = cfg.vision.clone()?;
    match VisionEncoder::new(v) {
        Ok(e) => {
            eprintln!(
                "[vision] images on (encoder starts with the first picture{})",
                if e.cfg.idle_secs > 0 {
                    format!(", stops after {} s without one", e.cfg.idle_secs)
                } else {
                    ", stops after each request's pictures".to_string()
                }
            );
            let arc = std::sync::Arc::new(e);
            {
                let w = std::sync::Arc::downgrade(&arc);
                let _ = std::thread::Builder::new().name("vision-idle".into()).spawn(move || loop {
                    std::thread::sleep(std::time::Duration::from_secs(5));
                    match w.upgrade() {
                        Some(e) => e.reap_idle(),
                        None => break,
                    }
                });
            }
            Some(arc)
        }
        Err(e) => {
            eprintln!("[vision] no encoder ({e}): text only");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_roundtrip_and_errors() {
        assert_eq!(base64_decode("aGVsbG8=").unwrap(), b"hello");
        assert_eq!(base64_decode("aGVs\nbG8").unwrap(), b"hello");
        assert!(base64_decode("aGVs*bG8").is_none());
    }

    #[test]
    fn content_key_differs_for_same_size() {
        assert_ne!(content_key(&[1, 2, 3, 4]), content_key(&[1, 2, 3, 5]));
        assert_ne!(content_key(&[0; 16]), content_key(&[0; 17]));
    }
}
