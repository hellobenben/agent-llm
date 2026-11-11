# agent-llm

A streaming chat-completion port, plus a client for OpenAI-compatible APIs.

The host passes prepared messages and tool definitions to `LLMProvider`. Text deltas arrive through `TextSink`. The call returns a complete `ModelTurn`. Sessions, tool execution, and persistence stay in the host.

Create one `OpenAIProvider` and share it. Its HTTP client is a connection pool. Each `generate` call is an independent request.

## Install

```toml
[dependencies]
agent-llm = "0.1"
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

Requires Rust edition 2024.

## One turn

`OpenAIProvider::new` takes an API key, a service base URL, and a model name. The base URL can be the official API or any compatible gateway.

Connecting gives up after 10 seconds. A read that receives no new bytes for 180 seconds fails. Bytes that keep arriving reset that timer, so a long reply is not cut off while it is still streaming.

```rust
use agent_llm::message::{Message, Role};
use agent_llm::openai::OpenAIProvider;
use agent_llm::types::{LLMProvider, TextSink};

struct PrintSink;

impl TextSink for PrintSink {
    fn on_delta(&self, text: &str) {
        print!("{text}");
    }
}

#[tokio::main]
async fn main() -> Result<(), agent_llm::message::Error> {
    let provider = OpenAIProvider::new(
        &std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY"),
        "https://api.openai.com/v1",
        "gpt-4.1-mini",
    );
    let turn = provider
        .generate(
            vec![Message {
                role: Role::User,
                content: "Introduce yourself in one sentence.".into(),
                tool_calls: vec![],
                tool_call_id: String::new(),
            }],
            vec![],
            Box::new(PrintSink),
        )
        .await?;
    println!("\n{}", turn.finish_reason.as_str());
    Ok(())
}
```

Deltas are delivered while the response streams. `ModelTurn.message.content` is the full text when `generate` returns.

## One response

`complete` returns the same `ModelTurn` and does not take a `TextSink`. `OpenAIProvider` sends one JSON request and does not set `stream`.

```rust
let turn = provider.complete(messages, tools).await?;
```

Another `LLMProvider` can leave `complete` as the default. That default calls `generate` and discards deltas. `OpenAIProvider` overrides it with the non-streaming endpoint. Both calls share the provider's connection pool and the same connect and read timeouts.

## Tool calls

A non-empty tool list is sent with `tool_choice: auto`. An empty list omits `tools`.

`ToolCall.arguments` is a JSON value. Argument fragments from the stream are joined and then parsed. Text that is not valid JSON is kept as a JSON string.

Send a tool result back by appending a `Role::Tool` message. Set `tool_call_id` to the call id and `content` to the result text.

## Your own provider

Depend on `LLMProvider`. An application that supplies its own model does not need `OpenAIProvider`.

```rust
use agent_llm::message::{Message, Role, ToolDefinition};
use agent_llm::types::{GenerateFuture, LLMProvider, ModelFinish, ModelTurn, TextSink};

struct Echo;

impl LLMProvider for Echo {
    fn generate(
        &self,
        messages: Vec<Message>,
        _tools: Vec<ToolDefinition>,
        sink: Box<dyn TextSink + Send>,
    ) -> GenerateFuture<'_> {
        Box::pin(async move {
            let text = messages
                .last()
                .map(|message| message.content.clone())
                .unwrap_or_default();
            sink.on_delta(&text);
            Ok(ModelTurn {
                message: Message {
                    role: Role::Assistant,
                    content: text,
                    tool_calls: vec![],
                    tool_call_id: String::new(),
                },
                finish_reason: ModelFinish::Stop,
                usage: None,
            })
        })
    }
}
```

## Types

| Module | Type | Role |
| --- | --- | --- |
| `agent_llm::types` | `LLMProvider` | Generates one turn |
| `agent_llm::types` | `TextSink` | Receives text deltas |
| `agent_llm::types` | `ModelTurn` | Full message, finish reason, and token usage |
| `agent_llm::types` | `ModelFinish` | `Stop`, `ToolCalls`, `Length`, or `ContentFilter` |
| `agent_llm::types` | `Usage` | `input_tokens` and `output_tokens` |
| `agent_llm::message` | `Message`, `Role`, `ToolCall` | Conversation sent to the model |
| `agent_llm::message` | `ToolDefinition` | Tool name, description, and JSON Schema |
| `agent_llm::message` | `Error` | Request failure, `Error::RequestError` |
| `agent_llm::openai` | `OpenAIProvider` | Streaming Chat Completions client |

`ModelFinish::as_str` returns `"stop"`, `"tool_calls"`, `"length"`, or `"content_filter"`.

`content_filter` means the provider stopped the turn with its own safety policy. The text may be empty or cut short. `length` means the output hit the length limit. When the stream omits a finish reason, the turn is `Stop` if there are no tool calls, and `ToolCalls` otherwise.

`Message` and `Role` serialize as JSON. Role values are `"system"`, `"user"`, `"assistant"`, and `"tool"`.
