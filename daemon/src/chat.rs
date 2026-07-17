use anyhow::{anyhow, Result};
use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer;

/// Chat template family, picked from the container's model_type on load.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Template {
    ChatML,
    Gemma,
}

impl Template {
    pub fn for_model_type(model_type: &str) -> Self {
        if model_type.contains("gemma") { Template::Gemma } else { Template::ChatML }
    }

    /// Generation stops (client-visible text is truncated) at any of these.
    pub fn stops(self) -> &'static [&'static str] {
        match self {
            Template::ChatML => &["<|im_end|>", "<|endoftext|>"],
            // Gemma 4 canonical: turns close with <turn|>; a thinking model
            // may also open a new channel — cut there too.
            Template::Gemma => &["<turn|>", "<|channel>", "<eos>"],
        }
    }
}

/// Everything generation needs that swaps together with the container on
/// /engine/load: the tokenizer and the chat template.
#[derive(Clone)]
pub struct ChatCtx {
    pub tok: Arc<Tok>,
    pub template: Template,
}

impl ChatCtx {
    /// Render messages with the container's template and append the
    /// assistant generation prefix.
    pub fn render<'a>(&self, messages: impl IntoIterator<Item = (&'a str, &'a str)>) -> String {
        match self.template {
            Template::ChatML => render_chatml(messages),
            Template::Gemma => render_gemma(messages),
        }
    }
}

pub fn render_chatml<'a>(messages: impl IntoIterator<Item = (&'a str, &'a str)>) -> String {
    let mut out = String::new();
    for (role, content) in messages {
        out.push_str("<|im_start|>");
        out.push_str(role);
        out.push('\n');
        out.push_str(content);
        out.push_str("<|im_end|>\n");
    }
    out.push_str("<|im_start|>assistant\n");
    out
}

/// Gemma 4 canonical template (chat_template.jinja, 2026-07-09): turns are
/// `<|turn>role\n...<turn|>`, roles are only user/model (system text rides as
/// a user turn), and the generation prompt opens an EMPTY thought channel
/// (`<|channel>thought\n<channel|>`) = thinking disabled, answer directly.
pub fn render_gemma<'a>(messages: impl IntoIterator<Item = (&'a str, &'a str)>) -> String {
    let mut out = String::from("<bos>");
    for (role, content) in messages {
        let role = if role == "assistant" { "model" } else { "user" };
        out.push_str("<|turn>");
        out.push_str(role);
        out.push('\n');
        out.push_str(content);
        out.push_str("<turn|>\n");
    }
    out.push_str("<|turn>model\n<|channel>thought\n<channel|>");
    out
}

/// Truncate `text` at the first stop string, if any. Returns the (possibly
/// shortened) text and whether a stop string was found.
pub fn strip_stop<'a>(text: &'a str, stops: &[&str]) -> (&'a str, bool) {
    let mut cut = None;
    for s in stops {
        if let Some(pos) = text.find(s) {
            cut = Some(cut.map_or(pos, |c: usize| c.min(pos)));
        }
    }
    match cut {
        Some(pos) => (&text[..pos], true),
        None => (text, false),
    }
}

pub struct Tok {
    inner: Tokenizer,
}

impl Tok {
    pub fn load(path: &Path) -> Result<Self> {
        let inner = Tokenizer::from_file(path)
            .map_err(|e| anyhow!("loading tokenizer {}: {e}", path.display()))?;
        Ok(Self { inner })
    }

    /// Encode WITHOUT special-token addition; the chat template supplies them.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow!("encode: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Decode keeping special tokens visible so stop strings can be found.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, false)
            .map_err(|e| anyhow!("decode: {e}"))
    }
}

/// Incremental decoder: feed one token id at a time, get back the newly
/// produced text. Decodes the full sequence each step and diffs against the
/// previously emitted prefix so multi-byte/multi-token characters render
/// correctly. Stops emitting once a stop string appears.
pub struct StreamDecoder {
    cc: ChatCtx,
    ids: Vec<u32>,
    prev: String,
    pub finished: bool,
}

impl StreamDecoder {
    pub fn new(cc: ChatCtx) -> Self {
        Self {
            cc,
            ids: Vec::new(),
            prev: String::new(),
            finished: false,
        }
    }

    pub fn push(&mut self, id: u32) -> Result<String> {
        if self.finished {
            return Ok(String::new());
        }
        self.ids.push(id);
        let full = self.cc.tok.decode(&self.ids)?;
        let (kept, stopped) = strip_stop(&full, self.cc.template.stops());
        let kept = kept.to_string();
        if stopped {
            self.finished = true;
        }
        let delta = kept
            .strip_prefix(self.prev.as_str())
            .unwrap_or("")
            .to_string();
        self.prev = kept;
        Ok(delta)
    }
}
