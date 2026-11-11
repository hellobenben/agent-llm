use async_openai::types::chat::{
    ChatCompletionMessageToolCall, ChatCompletionMessageToolCalls, ChatCompletionRequestAssistantMessage,
    ChatCompletionRequestAssistantMessageContent, ChatCompletionRequestMessage, ChatCompletionRequestSystemMessage,
    ChatCompletionRequestSystemMessageContent, ChatCompletionRequestToolMessage, ChatCompletionRequestToolMessageContent,
    ChatCompletionRequestUserMessage, ChatCompletionRequestUserMessageContent, ChatCompletionStreamOptions,
    ChatCompletionTool, ChatCompletionToolChoiceOption, ChatCompletionTools, CreateChatCompletionRequest, FinishReason,
    FunctionCall, FunctionObject, ToolChoiceOptions,
};
use async_openai::{Client, config::OpenAIConfig};
use futures::StreamExt;
use std::collections::BTreeMap;
use std::time::Duration;
use crate::message::{Error, Message, Role, ToolCall, ToolDefinition};
use crate::types::{GenerateFuture, LLMProvider, ModelFinish, ModelTurn, TextSink, Usage};

const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const READ_TIMEOUT: Duration = Duration::from_secs(180);

pub struct OpenAIProvider {
    client: Client<OpenAIConfig>,
    model: String,
}

impl OpenAIProvider {
    pub fn new(api_key: &str, base_url: &str, model: &str) -> Self {
        Self::with_timeouts(api_key, base_url, model, CONNECT_TIMEOUT, READ_TIMEOUT)
    }

    pub(crate) fn with_timeouts(
        api_key: &str,
        base_url: &str,
        model: &str,
        connect_timeout: Duration,
        read_timeout: Duration,
    ) -> Self {
        let cfg = OpenAIConfig::new().with_api_key(api_key).with_api_base(base_url);
        let http = reqwest::Client::builder()
            .connect_timeout(connect_timeout)
            .read_timeout(read_timeout)
            .build()
            .expect("openai http client");
        Self {
            client: Client::build(http, cfg),
            model: model.to_string(),
        }
    }
}

#[derive(Default)]
struct ToolAcc {
    id: String,
    name: String,
    arguments: String,
}

impl LLMProvider for OpenAIProvider {
    fn generate(
        &self,
        messages: Vec<Message>,
        tools: Vec<ToolDefinition>,
        sink: Box<dyn TextSink + Send>,
    ) -> GenerateFuture<'_> {
        let openai_msgs: Vec<_> = messages.iter().map(to_chat_msg).collect();
        let prepared = prepare_tools(tools);
        Box::pin(async move {
            let mut stream = self
                .client
                .chat()
                .create_stream(CreateChatCompletionRequest {
                    messages: openai_msgs,
                    model: self.model.clone(),
                    tools: prepared.tools,
                    tool_choice: prepared.tool_choice,
                    stream_options: Some(ChatCompletionStreamOptions {
                        include_usage: Some(true),
                        include_obfuscation: None,
                    }),
                    ..Default::default()
                })
                .await
                .map_err(|err| Error::RequestError { msg: err.to_string() })?;
            let mut text_buf = String::new();
            let mut acc: BTreeMap<u32, ToolAcc> = BTreeMap::new();
            let mut finish: Option<ModelFinish> = None;
            let mut usage: Option<Usage> = None;
            while let Some(chunk_res) = stream.next().await {
                let chunk = chunk_res.map_err(|err| Error::RequestError { msg: err.to_string() })?;
                if let Some(reported) = chunk.usage {
                    usage = Some(Usage {
                        input_tokens: reported.prompt_tokens,
                        output_tokens: reported.completion_tokens,
                    });
                }
                let Some(choice) = chunk.choices.into_iter().next() else {
                    continue;
                };
                if let Some(text) = choice.delta.content {
                    sink.on_delta(&text);
                    text_buf.push_str(&text);
                }
                for call in choice.delta.tool_calls.unwrap_or_default() {
                    let slot = acc.entry(call.index).or_default();
                    if let Some(id) = call.id {
                        slot.id = id;
                    }
                    if let Some(function) = call.function {
                        if let Some(name) = function.name {
                            slot.name = name;
                        }
                        if let Some(arguments) = function.arguments {
                            slot.arguments.push_str(&arguments);
                        }
                    }
                }
                if let Some(reason) = choice.finish_reason {
                    finish = Some(map_finish(reason));
                }
            }
            let tool_calls = acc
                .into_values()
                .map(|call| ToolCall {
                    id: call.id,
                    name: call.name,
                    arguments: parse_arguments(call.arguments),
                })
                .collect::<Vec<_>>();
            let finish_reason = finish_or_default(finish, &tool_calls);
            Ok(ModelTurn {
                message: Message {
                    role: Role::Assistant,
                    content: text_buf,
                    tool_calls,
                    tool_call_id: String::new(),
                },
                finish_reason,
                usage,
            })
        })
    }

    fn complete(&self, messages: Vec<Message>, tools: Vec<ToolDefinition>) -> GenerateFuture<'_> {
        let openai_msgs: Vec<_> = messages.iter().map(to_chat_msg).collect();
        let prepared = prepare_tools(tools);
        Box::pin(async move {
            let response = self
                .client
                .chat()
                .create(CreateChatCompletionRequest {
                    messages: openai_msgs,
                    model: self.model.clone(),
                    tools: prepared.tools,
                    tool_choice: prepared.tool_choice,
                    ..Default::default()
                })
                .await
                .map_err(|err| Error::RequestError { msg: err.to_string() })?;
            let Some(choice) = response.choices.into_iter().next() else {
                return Err(Error::RequestError { msg: "response has no choices".into() });
            };
            let tool_calls = choice
                .message
                .tool_calls
                .unwrap_or_default()
                .into_iter()
                .map(tool_call_from_response)
                .collect::<Vec<_>>();
            let usage = response.usage.map(|usage| Usage {
                input_tokens: usage.prompt_tokens,
                output_tokens: usage.completion_tokens,
            });
            let finish_reason = finish_or_default(choice.finish_reason.map(map_finish), &tool_calls);
            Ok(ModelTurn {
                message: Message {
                    role: Role::Assistant,
                    content: choice.message.content.unwrap_or_default(),
                    tool_calls,
                    tool_call_id: String::new(),
                },
                finish_reason,
                usage,
            })
        })
    }
}

struct PreparedTools {
    tools: Option<Vec<ChatCompletionTools>>,
    tool_choice: Option<ChatCompletionToolChoiceOption>,
}

fn prepare_tools(tools: Vec<ToolDefinition>) -> PreparedTools {
    let openai_tools: Vec<_> = tools
        .into_iter()
        .map(|tool| {
            ChatCompletionTools::Function(ChatCompletionTool {
                function: FunctionObject {
                    name: tool.name,
                    description: Some(tool.description),
                    parameters: Some(tool.input_schema),
                    strict: None,
                },
            })
        })
        .collect();
    let tool_choice = (!openai_tools.is_empty())
        .then_some(ChatCompletionToolChoiceOption::Mode(ToolChoiceOptions::Auto));
    let tools = (!openai_tools.is_empty()).then_some(openai_tools);
    PreparedTools { tools, tool_choice }
}

fn parse_arguments(raw: String) -> serde_json::Value {
    serde_json::from_str(&raw).unwrap_or(serde_json::Value::String(raw))
}

fn tool_call_from_response(call: ChatCompletionMessageToolCalls) -> ToolCall {
    match call {
        ChatCompletionMessageToolCalls::Function(call) => ToolCall {
            id: call.id,
            name: call.function.name,
            arguments: parse_arguments(call.function.arguments),
        },
        ChatCompletionMessageToolCalls::Custom(call) => ToolCall {
            id: call.id,
            name: call.custom_tool.name,
            arguments: serde_json::Value::String(call.custom_tool.input),
        },
    }
}

fn finish_or_default(finish: Option<ModelFinish>, tool_calls: &[ToolCall]) -> ModelFinish {
    finish.unwrap_or(if tool_calls.is_empty() {
        ModelFinish::Stop
    } else {
        ModelFinish::ToolCalls
    })
}

fn map_finish(reason: FinishReason) -> ModelFinish {
    match reason {
        FinishReason::Length => ModelFinish::Length,
        FinishReason::ToolCalls | FinishReason::FunctionCall => ModelFinish::ToolCalls,
        FinishReason::Stop  => ModelFinish::Stop,
        FinishReason::ContentFilter => ModelFinish::ContentFilter,
    }
}

#[allow(deprecated)]
pub(crate) fn to_chat_msg(msg: &Message) -> ChatCompletionRequestMessage {
    match msg.role {
        Role::System => ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
            content: ChatCompletionRequestSystemMessageContent::Text(msg.content.clone()),
            name: None,
        }),
        Role::User => ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
            content: ChatCompletionRequestUserMessageContent::Text(msg.content.clone()),
            name: None,
        }),
        Role::Assistant => {
            let calls: Vec<_> = msg
                .tool_calls
                .iter()
                .map(|call| {
                    ChatCompletionMessageToolCalls::Function(ChatCompletionMessageToolCall {
                        id: call.id.clone(),
                        function: FunctionCall {
                            name: call.name.clone(),
                            arguments: serde_json::to_string(&call.arguments).unwrap_or_else(|_| "{}".to_string()),
                        },
                    })
                })
                .collect();
            let tool_calls = if calls.is_empty() { None } else { Some(calls) };
            ChatCompletionRequestMessage::Assistant(ChatCompletionRequestAssistantMessage {
                content: Some(ChatCompletionRequestAssistantMessageContent::Text(msg.content.clone())),
                refusal: None,
                name: None,
                audio: None,
                tool_calls,
                function_call: None,
            })
        }
        Role::Tool => ChatCompletionRequestMessage::Tool(ChatCompletionRequestToolMessage {
            content: ChatCompletionRequestToolMessageContent::Text(msg.content.clone()),
            tool_call_id: msg.tool_call_id.clone(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::OpenAIProvider;
    use crate::message::{Message, Role, ToolCall};
    use crate::types::{LLMProvider, ModelFinish, ModelTurn, TextSink, Usage};
    use serde_json::{Value, json};
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    struct Collect(Arc<Mutex<String>>);

    impl TextSink for Collect {
        fn on_delta(&self, text: &str) {
            self.0.lock().unwrap().push_str(text);
        }
    }

    struct Completed {
        request: String,
        deltas: String,
        turn: ModelTurn,
    }

    fn complete(messages: Vec<Message>, chunks: &[Value]) -> Completed {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let payload = chunks
            .iter()
            .map(|chunk| format!("data: {chunk}\n\n"))
            .chain(std::iter::once("data: [DONE]\n\n".to_string()))
            .collect::<String>();
        let incoming = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
            let mut buf = Vec::new();
            let mut tmp = [0u8; 4096];
            let header_end = loop {
                let count = stream.read(&mut tmp).unwrap();
                assert!(count > 0, "vendor request closed before headers");
                buf.extend_from_slice(&tmp[..count]);
                if let Some(pos) = buf.windows(4).position(|window| window == b"\r\n\r\n") {
                    break pos + 4;
                }
            };
            let headers = String::from_utf8_lossy(&buf[..header_end]).to_string();
            let length = headers
                .lines()
                .find(|line| line.to_ascii_lowercase().starts_with("content-length:"))
                .and_then(|line| line.split(':').nth(1))
                .and_then(|value| value.trim().parse::<usize>().ok())
                .unwrap_or(0);
            while buf.len() < header_end + length {
                let count = stream.read(&mut tmp).unwrap();
                if count == 0 {
                    break;
                }
                buf.extend_from_slice(&tmp[..count]);
            }
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{payload}",
                payload.len()
            );
            stream.write_all(response.as_bytes()).unwrap();
            String::from_utf8_lossy(&buf[header_end..]).to_string()
        });

        let provider = OpenAIProvider::new("test-key", &format!("http://{address}/v1"), "gpt-test");
        let deltas = Arc::new(Mutex::new(String::new()));
        let sink = Box::new(Collect(Arc::clone(&deltas)));
        let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let turn = runtime.block_on(provider.generate(messages, vec![], sink)).unwrap();
        Completed {
            request: incoming.join().unwrap(),
            deltas: deltas.lock().unwrap().clone(),
            turn,
        }
    }

    fn user(content: &str) -> Message {
        Message {
            role: Role::User,
            content: content.to_string(),
            tool_calls: vec![],
            tool_call_id: String::new(),
        }
    }

    fn chunk(delta: Value, finish: Option<&str>, usage: Option<Value>) -> Value {
        json!({
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-test",
            "service_tier": null,
            "system_fingerprint": null,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish,
                "logprobs": null
            }],
            "usage": usage
        })
    }

    fn delta(content: Option<&str>, tool_calls: Option<Value>) -> Value {
        json!({
            "content": content,
            "function_call": null,
            "tool_calls": tool_calls,
            "role": null,
            "refusal": null
        })
    }

    #[test]
    fn stalled_connection_returns_before_the_server_closes() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
            let mut buf = [0u8; 4096];
            let _ = stream.read(&mut buf);
            std::thread::sleep(Duration::from_secs(5));
        });

        let provider = OpenAIProvider::with_timeouts(
            "test-key",
            &format!("http://{address}/v1"),
            "gpt-test",
            Duration::from_millis(200),
            Duration::from_millis(200),
        );
        let deltas = Arc::new(Mutex::new(String::new()));
        let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let started = std::time::Instant::now();
        let result = runtime.block_on(provider.generate(vec![user("hi")], vec![], Box::new(Collect(deltas))));
        assert!(result.is_err(), "{result:?}");
        assert!(started.elapsed() < Duration::from_secs(2), "waited {:?}", started.elapsed());
    }

    #[test]
    fn completion_reads_one_json_response_without_a_stream() {
        let body = json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-test",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "hello",
                    "tool_calls": [{
                        "id": "call-a",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{\"city\":\"Shanghai\"}"}
                    }]
                },
                "finish_reason": "tool_calls",
                "logprobs": null
            }],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
        });
        let payload = body.to_string();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let incoming = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
            let mut buf = Vec::new();
            let mut tmp = [0u8; 4096];
            let header_end = loop {
                let count = stream.read(&mut tmp).unwrap();
                assert!(count > 0);
                buf.extend_from_slice(&tmp[..count]);
                if let Some(pos) = buf.windows(4).position(|window| window == b"\r\n\r\n") {
                    break pos + 4;
                }
            };
            let headers = String::from_utf8_lossy(&buf[..header_end]).to_string();
            let length = headers
                .lines()
                .find(|line| line.to_ascii_lowercase().starts_with("content-length:"))
                .and_then(|line| line.split(':').nth(1))
                .and_then(|value| value.trim().parse::<usize>().ok())
                .unwrap_or(0);
            while buf.len() < header_end + length {
                let count = stream.read(&mut tmp).unwrap();
                if count == 0 {
                    break;
                }
                buf.extend_from_slice(&tmp[..count]);
            }
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{payload}",
                payload.len()
            );
            stream.write_all(response.as_bytes()).unwrap();
            String::from_utf8_lossy(&buf[header_end..]).to_string()
        });

        let provider = OpenAIProvider::new("test-key", &format!("http://{address}/v1"), "gpt-test");
        let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let turn = runtime
            .block_on(provider.complete(vec![user("hi")], vec![]))
            .unwrap();
        let request = incoming.join().unwrap();
        assert!(!request.contains("\"stream\""), "{request}");
        assert_eq!(turn.message.content, "hello");
        assert_eq!(turn.finish_reason, ModelFinish::ToolCalls);
        assert_eq!(turn.message.tool_calls.len(), 1);
        assert_eq!(turn.message.tool_calls[0].id, "call-a");
        assert_eq!(turn.message.tool_calls[0].name, "lookup");
        assert_eq!(turn.message.tool_calls[0].arguments, json!({"city": "Shanghai"}));
        assert_eq!(turn.usage, Some(Usage { input_tokens: 3, output_tokens: 2 }));
    }

    #[test]
    fn text_completion_returns_stop_and_streams_the_text() {
        let done = complete(
            vec![user("hi")],
            &[
                chunk(delta(Some("hello"), None), None, None),
                chunk(delta(None, None), Some("stop"), None),
            ],
        );
        assert_eq!(done.deltas, "hello");
        assert_eq!(done.turn.message.content, "hello");
        assert!(done.turn.message.tool_calls.is_empty());
        assert_eq!(done.turn.finish_reason, ModelFinish::Stop);
        assert!(done.turn.usage.is_none());
        assert!(done.request.contains("\"include_usage\":true"), "{}", done.request);
    }

    #[test]
    fn tool_call_completion_returns_both_calls() {
        let first = json!([{
            "index": 0,
            "id": "call-a",
            "type": "function",
            "function": {"name": "lookup", "arguments": ""}
        }]);
        let first_args = json!([{
            "index": 0,
            "function": {"arguments": "{\"city\":\"Shanghai\"}"}
        }]);
        let second = json!([{
            "index": 1,
            "id": "call-b",
            "type": "function",
            "function": {"name": "summarize", "arguments": "{\"ok\":true}"}
        }]);
        let done = complete(
            vec![user("hi")],
            &[
                chunk(delta(None, Some(first)), None, None),
                chunk(delta(None, Some(first_args)), None, None),
                chunk(delta(None, Some(second)), Some("tool_calls"), None),
            ],
        );
        assert!(done.deltas.is_empty());
        assert_eq!(done.turn.finish_reason, ModelFinish::ToolCalls);
        assert_eq!(done.turn.message.tool_calls.len(), 2);
        assert_eq!(done.turn.message.tool_calls[0].id, "call-a");
        assert_eq!(done.turn.message.tool_calls[0].name, "lookup");
        assert_eq!(done.turn.message.tool_calls[0].arguments, json!({"city": "Shanghai"}));
        assert_eq!(done.turn.message.tool_calls[1].id, "call-b");
        assert_eq!(done.turn.message.tool_calls[1].name, "summarize");
    }

    #[test]
    fn usage_chunk_without_choices_is_kept() {
        let usage = json!({
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-test",
            "service_tier": null,
            "system_fingerprint": null,
            "choices": [],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 4,
                "total_tokens": 16
            }
        });
        let done = complete(
            vec![user("hi")],
            &[chunk(delta(Some("hello"), None), Some("stop"), None), usage],
        );
        assert_eq!(
            done.turn.usage,
            Some(Usage { input_tokens: 12, output_tokens: 4 })
        );
    }

    #[test]
    fn omitted_usage_stays_empty() {
        let done = complete(vec![user("hi")], &[chunk(delta(Some("hello"), None), Some("stop"), None)]);
        assert!(done.turn.usage.is_none());
    }

    #[test]
    fn tool_message_is_sent_with_its_call_id() {
        let history = vec![
            user("hi"),
            Message {
                role: Role::Assistant,
                content: String::new(),
                tool_calls: vec![ToolCall {
                    id: "call-1".into(),
                    name: "lookup".into(),
                    arguments: json!({"city": "Shanghai"}),
                }],
                tool_call_id: String::new(),
            },
            Message {
                role: Role::Tool,
                content: r#"{"n":1}"#.into(),
                tool_calls: vec![],
                tool_call_id: "call-1".into(),
            },
        ];
        let done = complete(history, &[chunk(delta(Some("ok"), None), Some("stop"), None)]);
        let request: Value = serde_json::from_str(&done.request).unwrap();
        let messages = request["messages"].as_array().unwrap();
        let assistant = &messages[1];
        assert_eq!(assistant["role"], "assistant");
        assert_eq!(assistant["tool_calls"][0]["id"], "call-1");
        assert_eq!(assistant["tool_calls"][0]["function"]["name"], "lookup");
        let tool = &messages[2];
        assert_eq!(tool["role"], "tool");
        assert_eq!(tool["tool_call_id"], "call-1");
        assert_eq!(tool["content"], r#"{"n":1}"#);
    }
}
