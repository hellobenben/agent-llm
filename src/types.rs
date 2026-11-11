use std::pin::Pin;
use crate::message::{Error, Message, ToolDefinition};

pub trait TextSink: Send {
    fn on_delta(&self, text: &str);
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFinish {
    Stop,
    ToolCalls,
    Length,
    ContentFilter,
}

impl ModelFinish {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::ToolCalls => "tool_calls",
            Self::Length => "length",
            Self::ContentFilter => "content_filter",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelTurn {
    pub message: Message,
    pub finish_reason: ModelFinish,
    pub usage: Option<Usage>,
}

pub type GenerateFuture<'a> = Pin<Box<dyn Future<Output = Result<ModelTurn, Error>> + Send + 'a>>;

pub trait LLMProvider: Send + Sync {
    fn generate(
        &self,
        messages: Vec<Message>,
        tools: Vec<ToolDefinition>,
        sink: Box<dyn TextSink + Send>,
    ) -> GenerateFuture<'_>;

    fn complete(&self, messages: Vec<Message>, tools: Vec<ToolDefinition>) -> GenerateFuture<'_>;
}

struct Discard;

impl TextSink for Discard {
    fn on_delta(&self, _text: &str) {}
}