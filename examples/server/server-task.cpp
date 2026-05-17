#include "server-task.h"
#include "server-chat.h"

#include <cstdio>
#include <cstring>
#include <filesystem>

json result_timings::to_json() const {
    json base = {
        {"prompt_n",               prompt_n},
        {"prompt_ms",              prompt_ms},
        {"prompt_per_token_ms",    prompt_per_token_ms},
        {"prompt_per_second",      prompt_per_second},

        {"predicted_n",            predicted_n},
        {"predicted_ms",           predicted_ms},
        {"predicted_per_token_ms", predicted_per_token_ms},
        {"predicted_per_second",   predicted_per_second},

        {"n_ctx",           n_ctx},
        {"n_past",           n_past},
    };

    if (draft_n > 0) {
        base["draft_n"] = draft_n;
        base["draft_n_accepted"] = draft_n_accepted;
    }

    return base;
}


//json server_task_result_cmpl_partial::to_json_non_oaicompat_partial() {
//    // non-OAI-compat JSON
//    json res = json{
//        {"index",            index},
//        {"content",          content},
//        {"tokens",           tokens},
//        {"stop",             false},
//        {"id_slot",          id_multi},
//        {"tokens_predicted", n_decoded},
//        {"tokens_evaluated", n_prompt_tokens},
//    };
//    // populate the timings object when needed (usually for the last response or with timings_per_token enabled)
//    if (timings.prompt_n > 0) {
//        res.push_back({ "timings", timings.to_json() });
//    }
//    if (!probs_output.empty()) {
//        res["completion_probabilities"] = completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs);
//    }
//    return res;
//}

//json server_task_result_cmpl_final::to_json_non_oaicompat_final() {
//    json res = json{
//        {"index",               index},
//        {"content",             stream ? "" : content}, // in stream mode, content is already in last partial chunk
//        {"tokens",              stream ? std::vector<llama_token> {} : tokens},
//        {"id_slot",             id_multi},
//        {"stop",                true},
//        {"model",               oaicompat_model},
//        {"tokens_predicted",    n_decoded},
//        {"tokens_evaluated",    n_prompt_tokens},
//        //{"generation_settings", default_generation_settings_for_props.to_json()},
//        {"prompt",              prompt},
//        {"has_new_line",        has_new_line},
//        {"truncated",           truncated},
//        //{"stop_type",           stop_type_to_str(STOP_TYPE_EOS)},
//        {"stopping_word",       stopping_word},
//        {"tokens_cached",       n_tokens_cached},
//        {"timings",             timings.to_json()},
//    };
//    if (!stream && !probs_output.empty()) {
//        res["completion_probabilities"] = completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs);
//    }
//    return response_fields.empty() ? res : json_get_nested_values(response_fields, res);
//}

json server_task_result_cmpl_partial::to_json_non_oaicompat_partial() {
    // non-OAI-compat JSON
    return data;
}

json server_task_result_cmpl_final::to_json_non_oaicompat_final() {
    // non-OAI-compat JSON
    return data;
}

json server_task_result_cmpl_partial::to_json_oaicompat_partial() {
    std::time_t t = std::time(0);
    json logprobs = json(nullptr); // OAI default to null
    if (probs_output.size() > 0) {
        logprobs = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
    }
    json res = json{
        {"choices",            json::array({
            json{
                {"text",          content},
                {"index",         index},
                {"logprobs",      logprobs},
                {"finish_reason", nullptr},
            }
        })},
        {"created",            t},
        {"model",              oaicompat_model},
        {"object",             "text_completion"},
        {"usage", json {
            {"completion_tokens", n_decoded},
            {"prompt_tokens",     n_prompt_tokens},
            {"total_tokens",      n_decoded + n_prompt_tokens}
        }},
        {"id",                 oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat_partial();
    }
    if (timings.prompt_n >= 0) {
        res.push_back({ "timings", timings.to_json() });
    }

    return res;
}

json server_task_result_cmpl_final::usage_json_oaicompat() {
    return json{
        {"completion_tokens", n_decoded},
        {"prompt_tokens",     n_prompt_tokens},
        {"total_tokens",      n_decoded + n_prompt_tokens},
        {"prompt_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
    };
}


json server_task_result_cmpl_final::to_json_oaicompat_final() {
    std::time_t t = std::time(0);
    json logprobs = json(nullptr); // OAI default to null
    if (!stream && probs_output.size() > 0) {
        logprobs = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
    }
    json finish_reason = "length";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = "stop";
    }
    json res = json{
        {"choices",            json::array({
            json{
                {"text",          stream ? "" : content}, // in stream mode, content is already in last partial chunk
                {"index",         index},
                {"logprobs",      logprobs},
                {"finish_reason", finish_reason},
            }
        })},
        {"created",            t},
        {"model",              oaicompat_model},
        {"object",             "text_completion"},
        {"usage",              usage_json_oaicompat()},
        {"id", oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat_final();
    }
    if (timings.prompt_n >= 0) {
        res.push_back({ "timings", timings.to_json() });
    }

    return res;
}

json server_task_result_cmpl_partial::to_json_oaicompat_chat_partial() {
    bool first = n_decoded == 1;
    std::time_t t = std::time(0);
    json choices;

    std::vector<json> deltas;
    auto add_delta = [&](const json& delta) {
        deltas.push_back({
            {"choices", json::array({
                json {
                    {"finish_reason", nullptr},
                    {"index", 0},
                    {"delta", delta},
                },
            })},
            {"created", t},
            {"id", oaicompat_cmpl_id},
            {"model", oaicompat_model},
            {"object", "chat.completion.chunk"},
            {"usage", json {
                {"completion_tokens", n_decoded},
                {"prompt_tokens",     n_prompt_tokens},
                {"total_tokens",      n_decoded + n_prompt_tokens},
            }},
            });
    };
    // We have to send an initial update to conform to openai behavior
    if (first) {
        add_delta({
            {"role", "assistant"},
            {"content", nullptr},
            });
    }

    for (const auto& diff : oaicompat_msg_diffs) {        
        add_delta(server_chat_msg_diff_to_json_oaicompat(diff));
    }

    if (!deltas.empty()) {
        GGML_ASSERT(deltas[deltas.size() - 1].at("choices").size() >= 1);

        if (probs_output.size() > 0) {
            deltas[deltas.size() - 1].at("choices").at(0)["logprobs"] = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
            };
        }

        if (timings.prompt_n >= 0) {
            deltas[deltas.size() - 1].push_back({ "timings", timings.to_json() });
        }
    }

    return deltas;
}

json server_task_result_cmpl_partial::to_json_oaicompat_resp_partial() {
    std::vector<json> events;

    if (n_decoded == 1) {
        events.push_back(json{
            {"event", "response.created"},
            {"data", json{
                {"type", "response.created"},
                {"response", json{
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
        events.push_back(json{
            {"event", "response.in_progress"},
            {"data", json{
                {"type", "response.in_progress"},
                {"response", json{
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
    }

    for (const auto& diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty()) {
            if (!oai_resp_thinking_block_started) {
                events.push_back(json{
                    {"event", "response.output_item.added"},
                    {"data", json{
                        {"type", "response.output_item.added"},
                        {"item", json{
                            {"id",                oai_resp_reasoning_id},
                            {"summary",           json::array()},
                            {"type",              "reasoning"},
                            {"content",           json::array()},
                            {"encrypted_content", ""},
                            {"status",            "in_progress"},
                        }},
                    }},
                });
                oai_resp_thinking_block_started = true;
            }
            events.push_back(json{
                {"event", "response.reasoning_text.delta"},
                {"data", json{
                    {"type",    "response.reasoning_text.delta"},
                    {"delta",   diff.reasoning_content_delta},
                    {"item_id", oai_resp_reasoning_id},
                }},
            });
        }

        if (!diff.content_delta.empty()) {
            if (!oai_resp_text_block_started) {
                events.push_back(json{
                    {"event", "response.output_item.added"},
                    {"data", json{
                        {"type", "response.output_item.added"},
                        {"item", json{
                            {"content", json::array()},
                            {"id",      oai_resp_message_id},
                            {"role",    "assistant"},
                            {"status",  "in_progress"},
                            {"type",    "message"},
                        }},
                    }},
                });
                events.push_back(json{
                    {"event", "response.content_part.added"},
                    {"data", json{
                        {"type",    "response.content_part.added"},
                        {"item_id", oai_resp_message_id},
                        {"part", json{
                            {"type", "output_text"},
                            {"text", ""},
                        }},
                    }},
                });
                oai_resp_text_block_started = true;
            }
            events.push_back(json{
                {"event", "response.output_text.delta"},
                {"data", json{
                    {"type",    "response.output_text.delta"},
                    {"item_id", oai_resp_message_id},
                    {"delta",   diff.content_delta},
                }},
            });
        }

        if (!diff.tool_call_delta.name.empty()) {
            events.push_back(json{
                {"event", "response.output_item.added"},
                {"data", json{
                    {"type",  "response.output_item.added"},
                    {"item", json{
                        {"arguments", ""},
                        {"call_id",   "fc_" + diff.tool_call_delta.id},
                        {"name",      diff.tool_call_delta.name},
                        {"type",      "function_call"},
                        {"status",    "in_progress"},
                    }},
                }},
            });
            oai_resp_fc_id = diff.tool_call_delta.id;
        }

        if (!diff.tool_call_delta.arguments.empty()) {
            events.push_back(json{
                {"event", "response.function_call_arguments.delta"},
                {"data", json{
                    {"type",    "response.function_call_arguments.delta"},
                    {"delta",   diff.tool_call_delta.arguments},
                    {"item_id", "fc_" + oai_resp_fc_id},
                }},
            });
        }
    }

    return events;
}

json server_task_result_cmpl_final::to_json_oaicompat_chat_final() {
    std::string finish_reason = "length";
    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    }
    else {
        msg.role = "assistant";
        msg.content = content;
    }
    if (stop) {
        finish_reason = msg.tool_calls.empty() ? "stop" : "tool_calls";
    }


    json choice{
        {"finish_reason", finish_reason},
        {"index", 0},
        {"message", msg.to_json_oaicompat()},
    };

    if (!stream && probs_output.size() > 0) {
        choice["logprobs"] = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
    }

    std::time_t t = std::time(0);

    json res = json{
        {"choices",            json::array({choice})},
        {"created",            t},
        {"model",              oaicompat_model},
        {"object",             "chat.completion"},
        {"usage",              usage_json_oaicompat()},
        {"id", oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat_final();
    }
    if (timings.prompt_n >= 0) {
        res.push_back({ "timings", timings.to_json() });
    }

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_chat_stream() {
    std::time_t t = std::time(0);
    std::string finish_reason = "length";
    if (stop) {
        //if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = oaicompat_msg.tool_calls.empty() ? "stop" : "tool_calls";
    }

    json deltas = json::array();
    for (const auto& diff : oaicompat_msg_diffs) {
        deltas.push_back({
            {"choices", json::array({
                json {
                    {"finish_reason", nullptr},
                    {"index", 0},
                    {"delta", server_chat_msg_diff_to_json_oaicompat(diff)},
                },
            })},
            {"created", t},
            {"id", oaicompat_cmpl_id},
            {"model", oaicompat_model},
            {"object", "chat.completion.chunk"},
            });
    }

    deltas.push_back({
        {"choices", json::array({
            json {
                {"finish_reason", finish_reason},
                {"index", 0},
                {"delta", json::object()},
            },
        })},
        {"created",            t},
        {"id",                 oaicompat_cmpl_id},
        {"model",              oaicompat_model},
        {"object",             "chat.completion.chunk"},
        });
    if (include_usage) {
        // OpenAI API spec for chat.completion.chunks specifies an empty `choices` array for the last chunk when including usage
        // https://platform.openai.com/docs/api-reference/chat_streaming/streaming#chat_streaming/streaming-choices
        deltas.push_back({
            {"choices", json::array()},
            {"created",            t},
            {"id",                 oaicompat_cmpl_id},
            {"model",              oaicompat_model},
            {"object",             "chat.completion.chunk"},
            {"usage",              usage_json_oaicompat()},
            });
    }
    if (timings.prompt_n >= 0) {
        deltas.back().push_back({ "timings", timings.to_json() });
    }
    // extra fields for debugging purposes
    if (verbose && !deltas.empty()) {
        deltas.front()["__verbose"] = to_json_non_oaicompat_final();
    }

    return deltas;
}

json server_task_result_cmpl_final::to_json_oaicompat_resp_final() {
    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    }
    else {
        msg.role = "assistant";
        msg.content = content;
    }

    std::vector<json> output;

    if (!msg.reasoning_content.empty()) {
        output.push_back(json{
            {"id",      oai_resp_reasoning_id},
            {"summary", json::array()},
            {"type",    "reasoning"},
            {"content", json::array({json{
                {"text", msg.reasoning_content},
                {"type", "reasoning_text"},
            }})},
            {"encrypted_content", ""},
            {"status",            "completed"},
        });
    }

    if (!msg.content.empty()) {
        output.push_back(json{
            {"content", json::array({json{
                {"type",        "output_text"},
                {"annotations", json::array()},
                {"logprobs",    json::array()},
                {"text",        msg.content},
            }})},
            {"id",     oai_resp_message_id},
            {"role",   msg.role},
            {"status", "completed"},
            {"type",   "message"},
        });
    }

    for (const auto& tool_call : oaicompat_msg.tool_calls) {
        output.push_back(json{
            {"type",      "function_call"},
            {"status",    "completed"},
            {"arguments", tool_call.arguments},
            {"call_id",   "fc_" + tool_call.id},
            {"name",      tool_call.name},
        });
    }

    std::time_t t = std::time(0);
    json res = {
        {"completed_at", t},
        {"created_at",   t},
        {"id",           oai_resp_id},
        {"model",        oaicompat_model},
        {"object",       "response"},
        {"output",       output},
        {"status",       "completed"},
        {"usage",        json {
            {"input_tokens",  n_prompt_tokens},
            {"output_tokens", n_decoded},
            {"total_tokens",  n_decoded + n_prompt_tokens},
            {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
        }},
    };

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_resp_stream() {
    std::vector<json> events;
    std::vector<json> output;

    if (!oaicompat_msg.reasoning_content.empty()) {
        const json output_item = json{
            {"id",      oai_resp_reasoning_id},
            {"summary", json::array()},
            {"type",    "reasoning"},
            {"content", json::array({json{
                {"text", oaicompat_msg.reasoning_content},
                {"type", "reasoning_text"},
            }})},
            {"encrypted_content", ""},
        };

        events.push_back(json{
            {"event", "response.output_item.done"},
            {"data", json{
                {"type", "response.output_item.done"},
                {"item", output_item},
            }},
        });
        output.push_back(output_item);
    }

    if (!oaicompat_msg.content.empty()) {
        events.push_back(json{
            {"event", "response.output_text.done"},
            {"data", json{
                {"type",    "response.output_text.done"},
                {"item_id", oai_resp_message_id},
                {"text",    oaicompat_msg.content},
            }},
        });

        const json content_part = {
            {"type",        "output_text"},
            {"annotations", json::array()},
            {"logprobs",    json::array()},
            {"text",        oaicompat_msg.content},
        };

        events.push_back(json{
            {"event", "response.content_part.done"},
            {"data", json{
                {"type",    "response.content_part.done"},
                {"item_id", oai_resp_message_id},
                {"part",    content_part},
            }},
        });

        const json output_item = {
            {"type",    "message"},
            {"status",  "completed"},
            {"id",      oai_resp_message_id},
            {"content", json::array({content_part})},
            {"role",    "assistant"},
        };

        events.push_back(json{
            {"event", "response.output_item.done"},
            {"data", json{
                {"type", "response.output_item.done"},
                {"item", output_item},
            }},
        });
        output.push_back(output_item);
    }

    for (const auto& tool_call : oaicompat_msg.tool_calls) {
        const json output_item = {
            {"type",      "function_call"},
            {"status",    "completed"},
            {"arguments", tool_call.arguments},
            {"call_id",   "fc_" + tool_call.id},
            {"name",      tool_call.name},
        };
        events.push_back(json{
            {"event", "response.output_item.done"},
            {"data", json{
                {"type", "response.output_item.done"},
                {"item", output_item},
            }},
        });
        output.push_back(output_item);
    }

    std::time_t t = std::time(0);
    events.push_back(json{
        {"event", "response.completed"},
        {"data", json{
            {"type", "response.completed"},
            {"response", json{
                {"id",         oai_resp_id},
                {"object",     "response"},
                {"created_at", t},
                {"status",     "completed"},
                {"model",      oaicompat_model},
                {"output",     output},
                {"usage",      json {
                    {"input_tokens",  n_prompt_tokens},
                    {"output_tokens", n_decoded},
                    {"total_tokens",  n_decoded + n_prompt_tokens},
                    {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
                }}
            }},
        }},
    });

    return events;
}

json server_task_result_cmpl_final::to_json_anthropic_final() {
    std::string stop_reason = "max_tokens";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        stop_reason = oaicompat_msg.tool_calls.empty() ? "end_turn" : "tool_use";
    }

    json content_blocks = json::array();

    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    }
    else {
        msg.role = "assistant";
        msg.content = content;
    }

    if (!msg.reasoning_content.empty()) {
        content_blocks.push_back({
            {"type", "thinking"},
            {"thinking", msg.reasoning_content},
            {"signature", ""}
        });
    }

    if (!msg.content.empty()) {
        content_blocks.push_back({
            {"type", "text"},
            {"text", msg.content}
            });
    }

    for (const auto& tool_call : msg.tool_calls) {
        json tool_use_block = {
            {"type", "tool_use"},
            {"id", tool_call.id},
            {"name", tool_call.name}
        };

        try {
            tool_use_block["input"] = json::parse(tool_call.arguments);
        }
        catch (const std::exception&) {
            tool_use_block["input"] = json::object();
        }

        content_blocks.push_back(tool_use_block);
    }

    json res = {
        {"id", oaicompat_cmpl_id},
        {"type", "message"},
        {"role", "assistant"},
        {"content", content_blocks},
        {"model", oaicompat_model},
        {"stop_reason", stop_reason},
        {"stop_sequence", stopping_word.empty() ? nullptr : json(stopping_word)},
        {"usage", {
            {"cache_read_input_tokens", n_prompt_tokens_cache},
            {"input_tokens", n_prompt_tokens - n_prompt_tokens_cache},
            {"output_tokens", n_decoded}
        }}
    };

    return res;
}

json server_task_result_cmpl_final::to_json_anthropic_stream() {
    json events = json::array();

    std::string stop_reason = "max_tokens";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        stop_reason = oaicompat_msg.tool_calls.empty() ? "end_turn" : "tool_use";
    }

    size_t num_tool_calls = oaicompat_msg.tool_calls.size();

    size_t thinking_block_index = 0;
    size_t text_block_index = anthropic_thinking_block_started ? 1 : 0;

    bool thinking_block_started = anthropic_thinking_block_started;
    bool text_block_started = anthropic_text_block_started;
    std::set<size_t> tool_calls_started;

    for (const auto& diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_block_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", thinking_block_index},
                        {"content_block", {
                            {"type", "thinking"},
                            {"thinking", ""}
                        }}
                    }}
                    });
                thinking_block_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", thinking_block_index},
                    {"delta", {
                        {"type", "thinking_delta"},
                        {"thinking", diff.reasoning_content_delta}
                    }}
                }}
                });
        }

        if (!diff.content_delta.empty()) {
            if (!text_block_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", text_block_index},
                        {"content_block", {
                            {"type", "text"},
                            {"text", ""}
                        }}
                    }}
                    });
                text_block_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", text_block_index},
                    {"delta", {
                        {"type", "text_delta"},
                        {"text", diff.content_delta}
                    }}
                }}
                });
        }

        if (diff.tool_call_index != std::string::npos) {
            size_t content_block_index = (thinking_block_started ? 1 : 0) + (text_block_started ? 1 : 0) + diff.tool_call_index;

            if (tool_calls_started.find(diff.tool_call_index) == tool_calls_started.end()) {
                const auto& full_tool_call = oaicompat_msg.tool_calls[diff.tool_call_index];

                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", content_block_index},
                        {"content_block", {
                            {"type", "tool_use"},
                            {"id", full_tool_call.id},
                            {"name", full_tool_call.name}
                        }}
                    }}
                    });
                tool_calls_started.insert(diff.tool_call_index);
            }

            if (!diff.tool_call_delta.arguments.empty()) {
                events.push_back({
                    {"event", "content_block_delta"},
                    {"data", {
                        {"type", "content_block_delta"},
                        {"index", content_block_index},
                        {"delta", {
                            {"type", "input_json_delta"},
                            {"partial_json", diff.tool_call_delta.arguments}
                        }}
                    }}
                    });
            }
        }
    }

    if (thinking_block_started) {
        events.push_back({
            {"event", "content_block_delta"},
            {"data", {
                {"type", "content_block_delta"},
                {"index", thinking_block_index},
                {"delta", {
                    {"type", "signature_delta"},
                    {"signature", ""}
                }}
            }}
            });
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", thinking_block_index}
            }}
            });
    }

    if (text_block_started) {
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", text_block_index}
            }}
            });
    }

    for (size_t i = 0; i < num_tool_calls; i++) {
        size_t content_block_index = (thinking_block_started ? 1 : 0) + (text_block_started ? 1 : 0) + i;
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", content_block_index}
            }}
            });
    }

    events.push_back({
        {"event", "message_delta"},
        {"data", {
            {"type", "message_delta"},
            {"delta", {
                {"stop_reason", stop_reason},
                {"stop_sequence", stopping_word.empty() ? nullptr : json(stopping_word)}
            }},
            {"usage", {
                {"output_tokens", n_decoded}
            }}
        }}
        });

    events.push_back({
        {"event", "message_stop"},
        {"data", {
            {"type", "message_stop"}
        }}
        });

    // extra fields for debugging purposes
    if (verbose && !events.empty()) {
        events.front()["data"]["__verbose"] = to_json_non_oaicompat_final();
    }
    // Don't add timings for Anthropic API (breaks spec compliance)
    if (oaicompat != OAICOMPAT_TYPE_ANTHROPIC && timings.prompt_n >= 0 && !events.empty()) {
        events.back()["data"]["timings"] = timings.to_json();
    }

    return events;
}

json server_task_result_cmpl_partial::to_json_anthropic_partial() {
    json events = json::array();
    bool first = n_decoded == 1;

    size_t thinking_block_index = 0;
    size_t text_block_index = anthropic_has_reasoning ? 1 : 0;

    bool thinking_started = anthropic_thinking_block_started;
    bool text_started = anthropic_text_block_started;

    if (first) {
        events.push_back({
            {"event", "message_start"},
            {"data", {
                {"type", "message_start"},
                {"message", {
                    {"id", oaicompat_cmpl_id},
                    {"type", "message"},
                    {"role", "assistant"},
                    {"content", json::array()},
                    {"model", oaicompat_model},
                    {"stop_reason", nullptr},
                    {"stop_sequence", nullptr},
                    {"usage", {
                        {"cache_read_input_tokens", n_prompt_tokens_cache},
                        {"input_tokens", n_prompt_tokens - n_prompt_tokens_cache},
                        {"output_tokens", 0}
                    }}
                }}
            }}
            });
    }

    for (const auto& diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", thinking_block_index},
                        {"content_block", {
                            {"type", "thinking"},
                            {"thinking", ""}
                        }}
                    }}
                    });
                thinking_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", thinking_block_index},
                    {"delta", {
                        {"type", "thinking_delta"},
                        {"thinking", diff.reasoning_content_delta}
                    }}
                }}
                });
        }

        if (!diff.content_delta.empty()) {
            if (!text_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", text_block_index},
                        {"content_block", {
                            {"type", "text"},
                            {"text", ""}
                        }}
                    }}
                    });
                text_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", text_block_index},
                    {"delta", {
                        {"type", "text_delta"},
                        {"text", diff.content_delta}
                    }}
                }}
                });
        }

        if (diff.tool_call_index != std::string::npos) {
            size_t content_block_index = (anthropic_has_reasoning ? 1 : 0) + (text_started ? 1 : 0) + diff.tool_call_index;

            if (!diff.tool_call_delta.name.empty()) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", content_block_index},
                        {"content_block", {
                            {"type", "tool_use"},
                            {"id", diff.tool_call_delta.id},
                            {"name", diff.tool_call_delta.name}
                        }}
                    }}
                    });
            }

            if (!diff.tool_call_delta.arguments.empty()) {
                events.push_back({
                    {"event", "content_block_delta"},
                    {"data", {
                        {"type", "content_block_delta"},
                        {"index", content_block_index},
                        {"delta", {
                            {"type", "input_json_delta"},
                            {"partial_json", diff.tool_call_delta.arguments}
                        }}
                    }}
                    });
            }
        }
    }

    if (verbose && !events.empty() && first) {
        events.front()["data"]["__verbose"] = to_json_non_oaicompat_partial();
    }

    if (timings.prompt_n >= 0 && !events.empty()) {
        events.back()["data"]["timings"] = timings.to_json();
    }

    //if (is_progress && !events.empty()) {
    //    events.back()["data"]["prompt_progress"] = progress.to_json();
    //}

    return events;
}


size_t server_prompt::size() const {
    size_t res = data.size();

    for (const auto& checkpoint : checkpoints) {
        res += checkpoint.size();
    }

    return res;
}

size_t server_prompt_cache::size() const {
    size_t res = 0;

    for (const auto& state : states) {
        res += state.size();
    }

    return res;
}

size_t server_prompt_cache::n_tokens() const {
    size_t res = 0;

    for (const auto& state : states) {
        res += state.n_tokens();
    }
    return res;

}

bool server_prompt_cache::load(server_prompt& prompt, const server_tokens& tokens_new, llama_context* ctx, int32_t id_slot) {
    if (states.empty()) {
        return true;
    }

    server_tokens tokens_new_ex;
    thinking_tokens current_think_tokens;
    bool tokens_new_ex_valid = false;

    server_tokens prompt_tokens_ex;
    // We don't have think_tokens for 'prompt' (the slot's current content), 
    // so we'll use the ones from the first state as a reasonable guess if needed,
    // but actually prompt_tokens_ex is just used for sim_best initialization.
    {
        const thinking_tokens baseline_think = states.front().think_tokens;
        if (baseline_think.exclude) {
            prompt_tokens_ex = prompt.tokens.get_tokens_exclude_think(ctx, baseline_think);
            tokens_new_ex = tokens_new.get_tokens_exclude_think(ctx, baseline_think);
            current_think_tokens = baseline_think;
            tokens_new_ex_valid = true;
        } else {
            prompt_tokens_ex = prompt.tokens.clone();
        }
    }

    const auto lcp_best_init = prompt_tokens_ex.get_common_prefix(ctx, tokens_new_ex_valid ? tokens_new_ex : tokens_new);
    float f_keep_best = prompt_tokens_ex.empty() ? 0.0f : float(lcp_best_init.second) / prompt_tokens_ex.size();
    float sim_best = prompt_tokens_ex.get_tokens_similarity(ctx, tokens_new_ex_valid ? tokens_new_ex : tokens_new, prompt.n_kept_prompt, prompt.n_discarded_prompt);

    LLAMA_LOG_INFO(" - looking for better prompt, base f_keep = %.3f, sim = %.3f, n_keep = %d, n_discarded_prompt = %d\n", f_keep_best, sim_best, prompt.n_kept_prompt, prompt.n_discarded_prompt);

    auto it_best = states.end();

    // find the most similar cached prompt, that would also preserve the most context
    for (auto it = states.begin(); it != states.end(); ++it) {
        // Quick check for perfect exact match
        const size_t n_exact = it->tokens.get_common_prefix_exact(tokens_new);
        if (n_exact == it->tokens.size() && n_exact == tokens_new.size()) {
            it_best = it;
            sim_best = 1.0f;
            f_keep_best = 1.0f;
            break;
        }

        // Update tokens_new_ex if this entry has different think_tokens
        const bool think_changed = it->think_tokens.begin != current_think_tokens.begin
                                   || it->think_tokens.end   != current_think_tokens.end;
        if (it->think_tokens.exclude && (!tokens_new_ex_valid || think_changed)) {
            tokens_new_ex = tokens_new.get_tokens_exclude_think(ctx, it->think_tokens);
            current_think_tokens = it->think_tokens;
            tokens_new_ex_valid = true;
        } else if (!it->think_tokens.exclude) {
            tokens_new_ex_valid = false;
        }

        const server_tokens& target_tokens_new = tokens_new_ex_valid ? tokens_new_ex : tokens_new;

        server_tokens tokens;
        if (it->think_tokens.exclude) {
            tokens = it->tokens.get_tokens_exclude_think(ctx, it->think_tokens);
        }
        else {
            tokens = it->tokens.clone();
        }

        const auto lcp_cur = tokens.get_common_prefix(ctx, target_tokens_new);
        const float f_keep_cur = tokens.empty() ? 0.0f : float(lcp_cur.first) / tokens.size();
        
        float sim_cur = 0.0f;
        if (it->n_kept_prompt == 0 && it->n_discarded_prompt == 0) {
            sim_cur = get_slot_similarity(lcp_cur.second, target_tokens_new.size(), tokens.size());
        } else {
            sim_cur = tokens.get_tokens_similarity(ctx, target_tokens_new, it->n_kept_prompt, it->n_discarded_prompt);
        }

        if (sim_best < sim_cur) {
            f_keep_best = f_keep_cur;
            sim_best = sim_cur;
            it_best = it;
            
            if (sim_best >= 1.0f) {
                break;
            }
        }
    }

    if (it_best != states.end()) {
        LLAMA_LOG_INFO(" - found better prompt with f_keep = %.3f, sim = %.3f, n_keep = %d, n_discarded_prompt = %d\n", f_keep_best, sim_best, it_best->n_kept_prompt, it_best->n_discarded_prompt);

        // Disk tier: if the chosen entry was demoted, page its blob back into
        // RAM before handing it to llama_state_seq_set_data.
        if (!it_best->disk_file.empty() && it_best->data.empty()) {
            if (!restore_from_disk(*it_best)) {
                LLAMA_LOG_INFO("failed to restore prompt from disk (%s)\n", it_best->disk_file.c_str());
                return false;
            }
        }

        const size_t size = it_best->data.size();
        const size_t n = llama_state_seq_set_data(ctx, it_best->data.data(), size, id_slot, 0);
        LLAMA_LOG_INFO(" - set_data restored %zu / %zu bytes\n", n, size);
        if (n != size) {
            LLAMA_LOG_INFO("failed to restore state with size %zu\n", size);
            return false;
        }

        it_best->data.clear();
        it_best->data.shrink_to_fit();

        prompt = std::move(*it_best);

        states.erase(it_best);
    }

    return true;
}

server_prompt* server_prompt_cache::alloc(const server_prompt& prompt, size_t state_size) {
    auto tokens_ctx_shift = prompt.tokens.clone();  // copy cache tokens
    tokens_ctx_shift.discard_n_tokens(prompt.n_kept_prompt, prompt.n_discarded_prompt);

    for (auto it = states.begin(); it != states.end();) {
        // first check if the current state is contained fully in the cache
        // Use exact matching first to skip expensive fuzzy matching
        const size_t n_exact = it->tokens.get_common_prefix_exact(tokens_ctx_shift);

        if (n_exact == tokens_ctx_shift.size()) {
            LLAMA_LOG_INFO("%s", " - prompt is already in the cache, skipping\n");
            return nullptr;
        }

        const auto prefix = it->tokens.get_common_prefix(ctx, tokens_ctx_shift);
        const size_t len = prefix.first;
        const size_t len_prompt = prefix.second;

        if (len_prompt == tokens_ctx_shift.size()) {
            LLAMA_LOG_INFO("%s", " - prompt is already in the cache, skipping\n");
            return nullptr;
        }
        // next, remove any cached prompts that are fully contained in the current prompt
        else if (len == it->tokens.size()) {
            LLAMA_LOG_INFO(" - removing obsolete cached prompt with length %d\n", (int)len);
            it = states.erase(it);
        }
        else {
            ++it;
        }
    }

    std::vector<uint8_t> state_data;

    // check if we can allocate enough memory for the new state
    try {
        state_data.resize(state_size);
    }
    catch (const std::bad_alloc& e) {
        LLAMA_LOG_INFO("failed to allocate memory for prompt cache state: %s\n", e.what());

        limit_size = std::max<size_t>(1, 0.4 * size());

        LLAMA_LOG_INFO(" - cache size limit reduced to %.3f MiB\n", limit_size / (1024.0 * 1024.0));

        update();

        return nullptr;
    }

    // TODO: for some reason we can't copy server_tokens, so we have to do this workaround
    auto& cur = states.emplace_back();
    cur = {
        /*.tokens          =*/ prompt.tokens.clone(),
        /*.n_keep          =*/ prompt.n_kept_prompt,
        /*.n_discarded_prompt     =*/ prompt.n_discarded_prompt,
        /*.think_tokens                   =*/ prompt.think_tokens,
        /*.data            =*/ std::move(state_data),
        /*.checkpoints     =*/ prompt.checkpoints,
    };

    // Propagate transient state from the source prompt so that an entry which
    // came back from disk earlier in this process is still recognized as
    // "continued" on the next demotion. Touch last_used_us so the cold demoter
    // gives a freshly stored entry its full idle budget.
    cur.was_restored_from_disk = prompt.was_restored_from_disk;
    cur.last_reason            = prompt.last_reason;
    cur.last_used_us           = ggml_time_us();

    return &cur;
}


void server_prompt_cache::update() {
    if (limit_size > 0) {
        // always keep at least one state, regardless of the limits
        while (states.size() > 1 && size() > limit_size) {
            if (states.empty()) {
                break;
            }

            // Disk-resident entries have size()=0 so they never cause the
            // size limit to be exceeded.  Skip them to find the oldest
            // RAM-resident entry that is actually consuming memory.
            auto victim_it = states.begin();
            while (victim_it != states.end() && !victim_it->disk_file.empty()) {
                ++victim_it;
            }
            if (victim_it == states.end()) {
                break; // all entries are on disk; nothing to free
            }

            auto & victim = *victim_it;

            // Disk tier: instead of dropping, demote to disk if enabled and
            // the victim is still RAM-resident and worth saving.
            const bool demote = !disk_dir.empty()
                                && victim.disk_file.empty()
                                && !victim.data.empty()
                                && (size_t)victim.n_tokens() >= n_min_disk;

            const auto reason = victim.was_restored_from_disk
                                ? disk_save_reason::continued
                                : disk_save_reason::evict;

            if (demote && dump_to_disk(victim, reason)) {
                LLAMA_LOG_INFO(" - cache size limit reached, demoting entry to disk (%s, %.3f MiB)\n",
                    victim.disk_file.c_str(), victim.data_disk_size / (1024.0 * 1024.0));
                // Move the now-disk-resident entry to the back so newer RAM
                // entries can be demoted next time without revisiting this one.
                states.splice(states.end(), states, victim_it);
                continue;
            }

            LLAMA_LOG_INFO(" - cache size limit reached, removing oldest RAM entry (size = %.3f MiB)\n",
                victim.size() / (1024.0 * 1024.0));

            states.erase(victim_it);
        }
    }

    // Proactive cold demotion: move RAM-resident entries idle for longer than
    // COLD_THRESHOLD_US to disk even when we are below limit_size. Keeps the
    // working set in RAM small and amortizes write cost away from the moment
    // RAM is actually exhausted. last_used_us == 0 means we never observed a
    // touch (e.g. fresh-from-bootstrap), so we skip those until they are used.
    if (!disk_dir.empty()) {
        constexpr int64_t COLD_THRESHOLD_US = 300LL * 1000LL * 1000LL; // 5 min
        const int64_t now = ggml_time_us();

        for (auto it = states.begin(); it != states.end(); ) {
            const bool ram_resident = !it->data.empty() && it->disk_file.empty();
            const bool eligible     = ram_resident
                                      && it->last_used_us > 0
                                      && (now - it->last_used_us) > COLD_THRESHOLD_US
                                      && (size_t) it->n_tokens() >= n_min_disk;

            if (eligible) {
                auto it_next = std::next(it);
                if (dump_to_disk(*it, disk_save_reason::cold)) {
                    LLAMA_LOG_INFO(" - cold demotion: idle %.1f s, moved to disk (%s)\n",
                        (now - it->last_used_us) / 1e6, it->disk_file.c_str());
                    states.splice(states.end(), states, it);
                }
                it = it_next;
            } else {
                ++it;
            }
        }
    }

    // Disk tier LRU (no-op when disk_dir is empty).
    if (!disk_dir.empty() && limit_disk_bytes > 0) {
        enforce_disk_limit();
    }

    // average size per token
    const float size_per_token = std::max<float>(1.0f, float(size()) / (std::max<size_t>(1, n_tokens())));

    // dynamically increase the token limit if it can fit in the memory limit
    const size_t limit_tokens_cur = limit_size > 0 ? std::max<size_t>(limit_tokens, limit_size / size_per_token) : limit_tokens;

    LLAMA_LOG_INFO(" - cache state: %zu prompts, %.3f MiB (limits: %.3f MiB, %zu tokens, %zu est)\n",
        states.size(), size() / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0), limit_tokens, limit_tokens_cur);

    for (const auto& state : states) {
        LLAMA_LOG_INFO("   - prompt %p: %7d tokens, %7d discarded, checkpoints: %2zu, %9.3f MiB\n",
            (const void*)&state, state.n_tokens(), state.n_discarded_prompt, state.checkpoints.size(), state.size() / (1024.0 * 1024.0));
    }
}

// ============================================================================
// Disk tier (Phase 1 scaffold — stubs only, no I/O logic yet).
//
// File layout for <disk_dir>/prompt-NNNNNNNN.kv (v2):
//   [ kv_file_header (40 B) ] [ token_count * u32 LE tokens ] [ data blob ]
//   [ kv_checkpoint_record (32 B) ] [ checkpoint blob ] × checkpoint_count
//
// Naming: prompt-<next_disk_id>.kv (monotonic counter, no SHA1).
// LRU on disk: mtime-based, applied by enforce_disk_limit().
//
// Reserved bits in the header allow adding ds4-style save-reason tagging
// (see PROJECT_CONTEXT.md "Idee da rubare a ds4 — Phase 1.5") without
// breaking the format.
// ============================================================================

namespace {

struct kv_file_header {
    char     magic[4];          // "IKKV"
    uint8_t  version;           // format version (current: 2)
    uint8_t  save_reason;       // 0=unknown; reserved for Phase 1.5
    uint8_t  checkpoint_count;  // v2+: number of checkpoint records after the data blob
    uint8_t  reserved;          // must be zero
    uint32_t token_count;
    uint32_t n_kept_prompt;
    uint32_t n_discarded_prompt;
    uint64_t data_size;
    uint64_t creation_time_us;
};
static_assert(sizeof(kv_file_header) == 40, "kv_file_header layout changed; update writers/readers");

// Each checkpoint record is written after the main data blob (v2+).
// Followed immediately by `data_size` bytes of raw checkpoint blob.
struct kv_checkpoint_record {
    int32_t  pos_min;
    int32_t  pos_max;
    int32_t  pos_min_prompt;
    int32_t  pos_max_prompt;
    int64_t  n_tokens;
    uint64_t data_size;
};
static_assert(sizeof(kv_checkpoint_record) == 32, "kv_checkpoint_record layout changed; update writers/readers");

constexpr char    KV_MAGIC[4] = {'I', 'K', 'K', 'V'};
constexpr uint8_t KV_VERSION  = 2;

} // namespace

bool server_prompt_cache::write_kv_file_atomic(const std::string & fname,
                                               const server_prompt & p,
                                               disk_save_reason reason) {
    if (disk_dir.empty()) return false;

    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(disk_dir, ec);
    if (ec) {
        LLAMA_LOG_INFO(" - disk write: cannot create %s: %s\n", disk_dir.c_str(), ec.message().c_str());
        return false;
    }

    const std::string fpath     = disk_dir + "/" + fname;
    const std::string fpath_tmp = fpath + ".tmp";

    const uint32_t token_count = (uint32_t) p.tokens.n_tokens();
    const uint64_t data_size   = (uint64_t) p.data.size();
    const uint8_t  chkpt_count = (uint8_t) std::min(p.checkpoints.size(), (size_t) 255);

    kv_file_header hdr{};
    std::memcpy(hdr.magic, KV_MAGIC, sizeof(hdr.magic));
    hdr.version            = KV_VERSION;
    hdr.save_reason        = (uint8_t) reason;
    hdr.checkpoint_count   = chkpt_count;
    hdr.token_count        = token_count;
    hdr.n_kept_prompt      = (uint32_t) p.n_kept_prompt;
    hdr.n_discarded_prompt = (uint32_t) p.n_discarded_prompt;
    hdr.data_size          = data_size;
    hdr.creation_time_us   = (uint64_t) ggml_time_us();

    FILE * f = std::fopen(fpath_tmp.c_str(), "wb");
    if (f == nullptr) {
        LLAMA_LOG_INFO(" - disk write: cannot open %s for write\n", fpath_tmp.c_str());
        return false;
    }

    bool ok = std::fwrite(&hdr, sizeof(hdr), 1, f) == 1;

    if (ok) {
        std::vector<uint32_t> tok_buf;
        tok_buf.reserve(token_count);
        for (uint32_t i = 0; i < token_count; ++i) {
            tok_buf.push_back((uint32_t) p.tokens[i]);
        }
        ok = std::fwrite(tok_buf.data(), sizeof(uint32_t), token_count, f) == token_count;
    }

    if (ok && data_size > 0) {
        ok = std::fwrite(p.data.data(), 1, data_size, f) == data_size;
    }

    if (ok) {
        size_t written = 0;
        for (const auto & chk : p.checkpoints) {
            if (written++ >= chkpt_count) break;
            kv_checkpoint_record rec{};
            rec.pos_min        = chk.pos_min;
            rec.pos_max        = chk.pos_max;
            rec.pos_min_prompt = chk.pos_min_prompt;
            rec.pos_max_prompt = chk.pos_max_prompt;
            rec.n_tokens       = chk.n_tokens;
            rec.data_size      = (uint64_t) chk.data.size();
            ok = std::fwrite(&rec, sizeof(rec), 1, f) == 1;
            if (ok && rec.data_size > 0) {
                ok = std::fwrite(chk.data.data(), 1, rec.data_size, f) == rec.data_size;
            }
            if (!ok) break;
        }
    }

    std::fclose(f);

    if (!ok) {
        std::remove(fpath_tmp.c_str());
        LLAMA_LOG_INFO(" - disk write: failed, removed partial %s\n", fpath_tmp.c_str());
        return false;
    }

    if (std::rename(fpath_tmp.c_str(), fpath.c_str()) != 0) {
        std::remove(fpath_tmp.c_str());
        LLAMA_LOG_INFO(" - disk write: rename %s -> %s failed\n", fpath_tmp.c_str(), fpath.c_str());
        return false;
    }

    return true;
}

bool server_prompt_cache::dump_to_disk(server_prompt & p, disk_save_reason reason) {
    if (p.tokens.has_mtmd_data()) {
        LLAMA_LOG_INFO("%s", " - skipping disk demotion: prompt contains multimodal chunks (unsupported)\n");
        return false;
    }
    if (disk_dir.empty() || p.data.empty() || p.tokens.n_tokens() <= 0) {
        return false;
    }

    const int64_t t_start = ggml_time_us();

    // Zero-padded counter so filenames sort lexicographically by insertion
    // order (matches mtime order and makes bootstrap counter-parsing trivial).
    char fname_buf[40];
    std::snprintf(fname_buf, sizeof(fname_buf), "prompt-%016llx.kv",
                  (unsigned long long) next_disk_id);
    const std::string fname = fname_buf;

    if (!write_kv_file_atomic(fname, p, reason)) {
        return false;
    }

    const uint32_t token_count = (uint32_t) p.tokens.n_tokens();
    const uint64_t data_size   = (uint64_t) p.data.size();
    const uint8_t  chkpt_count = (uint8_t) std::min(p.checkpoints.size(), (size_t) 255);

    p.disk_file      = fname;
    p.data_disk_size = data_size;
    p.last_reason    = (uint8_t) reason;
    p.data.clear();
    p.data.shrink_to_fit();
    p.checkpoints.clear();

    ++next_disk_id;

    LLAMA_LOG_INFO(" - disk dump: wrote %s (%u tokens, %u checkpoints, %.3f MiB, %.2f ms)\n",
                   fname.c_str(), token_count, (unsigned) chkpt_count,
                   data_size / (1024.0 * 1024.0),
                   (ggml_time_us() - t_start) / 1000.0);
    return true;
}

bool server_prompt_cache::live_checkpoint(int32_t slot_id, server_prompt & snap) {
    if (disk_dir.empty()) return false;
    if (snap.tokens.has_mtmd_data()) return false;
    if (snap.data.empty() || snap.tokens.n_tokens() <= 0) return false;
    if ((size_t) snap.tokens.n_tokens() < n_min_disk) return false;

    char fname_buf[32];
    std::snprintf(fname_buf, sizeof(fname_buf), "slot-%d.kv", (int) slot_id);
    const std::string fname = fname_buf;

    const int64_t t_start = ggml_time_us();
    const bool ok = write_kv_file_atomic(fname, snap, disk_save_reason::continued);
    if (!ok) {
        return false;
    }

    LLAMA_LOG_INFO(" - live checkpoint: wrote %s (%d tokens, %.3f MiB, %.2f ms)\n",
                   fname.c_str(), snap.tokens.n_tokens(),
                   snap.data.size() / (1024.0 * 1024.0),
                   (ggml_time_us() - t_start) / 1000.0);
    return true;
}

void server_prompt_cache::drop_live_checkpoint(int32_t slot_id) {
    if (disk_dir.empty()) return;
    char fname_buf[32];
    std::snprintf(fname_buf, sizeof(fname_buf), "slot-%d.kv", (int) slot_id);
    const std::string fpath = disk_dir + "/" + fname_buf;
    std::remove(fpath.c_str()); // best effort; silent if missing
}

bool server_prompt_cache::restore_from_disk(server_prompt & p) {
    if (disk_dir.empty() || p.disk_file.empty()) {
        return false;
    }

    const int64_t t_start = ggml_time_us();
    const std::string fpath = disk_dir + "/" + p.disk_file;

    FILE * f = std::fopen(fpath.c_str(), "rb");
    if (f == nullptr) {
        LLAMA_LOG_INFO(" - disk restore: cannot open %s\n", fpath.c_str());
        return false;
    }

    kv_file_header hdr{};
    if (std::fread(&hdr, sizeof(hdr), 1, f) != 1) {
        std::fclose(f);
        LLAMA_LOG_INFO(" - disk restore: failed to read header from %s\n", fpath.c_str());
        return false;
    }

    if (std::memcmp(hdr.magic, KV_MAGIC, sizeof(hdr.magic)) != 0 || hdr.version != KV_VERSION) {
        std::fclose(f);
        LLAMA_LOG_INFO(" - disk restore: invalid magic/version in %s\n", fpath.c_str());
        return false;
    }

    if (hdr.token_count != (uint32_t)p.tokens.n_tokens() || hdr.data_size != p.data_disk_size) {
        std::fclose(f);
        LLAMA_LOG_INFO(" - disk restore: token/size mismatch in %s (expected %u/%llu, got %u/%llu)\n", 
                       fpath.c_str(), (uint32_t)p.tokens.n_tokens(), (unsigned long long)p.data_disk_size, 
                       hdr.token_count, (unsigned long long)hdr.data_size);
        return false;
    }

    if (std::fseek(f, hdr.token_count * sizeof(uint32_t), SEEK_CUR) != 0) {
        std::fclose(f);
        LLAMA_LOG_INFO(" - disk restore: failed to seek past tokens in %s\n", fpath.c_str());
        return false;
    }

    p.data.resize(hdr.data_size);
    if (hdr.data_size > 0) {
        if (std::fread(p.data.data(), 1, hdr.data_size, f) != hdr.data_size) {
            p.data.clear();
            std::fclose(f);
            LLAMA_LOG_INFO(" - disk restore: failed to read blob from %s\n", fpath.c_str());
            return false;
        }
    }

    // Deserialize checkpoints (v2).
    p.checkpoints.clear();
    for (uint8_t i = 0; i < hdr.checkpoint_count; ++i) {
        kv_checkpoint_record rec{};
        if (std::fread(&rec, sizeof(rec), 1, f) != 1) {
            p.data.clear();
            p.checkpoints.clear();
            std::fclose(f);
            LLAMA_LOG_INFO(" - disk restore: failed to read checkpoint record %u from %s\n", (unsigned)i, fpath.c_str());
            return false;
        }
        server_prompt_checkpoint chk{};
        chk.pos_min        = rec.pos_min;
        chk.pos_max        = rec.pos_max;
        chk.pos_min_prompt = rec.pos_min_prompt;
        chk.pos_max_prompt = rec.pos_max_prompt;
        chk.n_tokens       = rec.n_tokens;
        if (rec.data_size > 0) {
            chk.data.resize(rec.data_size);
            if (std::fread(chk.data.data(), 1, rec.data_size, f) != rec.data_size) {
                p.data.clear();
                p.checkpoints.clear();
                std::fclose(f);
                LLAMA_LOG_INFO(" - disk restore: failed to read checkpoint data %u from %s\n", (unsigned)i, fpath.c_str());
                return false;
            }
        }
        p.checkpoints.push_back(std::move(chk));
    }

    p.n_kept_prompt      = (int) hdr.n_kept_prompt;
    p.n_discarded_prompt = (int) hdr.n_discarded_prompt;

    std::fclose(f);

    if (std::remove(fpath.c_str()) != 0) {
        LLAMA_LOG_INFO(" - disk restore: warning, failed to remove %s\n", fpath.c_str());
    }

    const char * reason_str = "unknown";
    switch (hdr.save_reason) {
        case 1: reason_str = "cold";      break;
        case 2: reason_str = "continued"; break;
        case 3: reason_str = "evict";     break;
        case 4: reason_str = "shutdown";  break;
        default: break;
    }
    LLAMA_LOG_INFO(" - disk restore: loaded %s (%u tokens, %u checkpoints, %.3f MiB, save_reason=%s, %.2f ms)\n",
                   p.disk_file.c_str(), hdr.token_count, (unsigned) hdr.checkpoint_count,
                   hdr.data_size / (1024.0 * 1024.0), reason_str,
                   (ggml_time_us() - t_start) / 1000.0);

    p.disk_file.clear();
    p.data_disk_size = 0;
    p.last_reason             = hdr.save_reason;
    p.was_restored_from_disk  = true;
    p.last_used_us            = ggml_time_us();

    return true;
}

size_t server_prompt_cache::flush_all_to_disk(disk_save_reason reason) {
    if (disk_dir.empty()) return 0;

    size_t demoted = 0;
    for (auto & p : states) {
        const bool ram_resident = !p.data.empty() && p.disk_file.empty();
        if (!ram_resident) continue;
        if ((size_t) p.n_tokens() < n_min_disk) continue;

        if (dump_to_disk(p, reason)) {
            ++demoted;
        }
    }

    if (demoted > 0) {
        LLAMA_LOG_INFO(" - disk flush: demoted %zu RAM entries (reason=%u)\n",
                       demoted, (unsigned) reason);
    }
    return demoted;
}

void server_prompt_cache::enforce_disk_limit() {
    if (disk_dir.empty() || limit_disk_bytes == 0) return;

    uint64_t total_disk_size = 0;
    for (const auto& p : states) {
        total_disk_size += p.data_disk_size;
    }

    if (total_disk_size <= limit_disk_bytes) return;

    // Eviction priority by save_reason: cold and unknown go first (least
    // valuable), then evict (RAM-pressure demotions), then shutdown (clean
    // exit; keep when possible), then continued (entries observed in active
    // use this session; kept longest). Within each class we still iterate in
    // list order, which approximates mtime-LRU.
    using R = disk_save_reason;
    const R order[] = { R::cold, R::unknown, R::evict, R::shutdown, R::continued };

    for (const R target : order) {
        if (total_disk_size <= limit_disk_bytes) break;

        auto it = states.begin();
        while (it != states.end() && total_disk_size > limit_disk_bytes) {
            const bool on_disk = !it->disk_file.empty();
            const bool match   = (R) it->last_reason == target;
            if (on_disk && match) {
                std::string fpath = disk_dir + "/" + it->disk_file;
                if (std::remove(fpath.c_str()) == 0) {
                    LLAMA_LOG_INFO(" - disk lru: unlinked %s (reason=%u, freed %.3f MiB)\n",
                                   it->disk_file.c_str(), (unsigned) it->last_reason,
                                   it->data_disk_size / (1024.0 * 1024.0));
                } else {
                    LLAMA_LOG_INFO(" - disk lru: warning, failed to unlink %s\n", it->disk_file.c_str());
                }
                total_disk_size -= it->data_disk_size;
                it = states.erase(it);
            } else {
                ++it;
            }
        }
    }
}

void server_prompt_cache::bootstrap_disk_tier() {
    if (disk_dir.empty()) return;

    namespace fs = std::filesystem;
    std::error_code ec;

    if (!fs::exists(disk_dir, ec)) {
        fs::create_directories(disk_dir, ec);
        return;
    }

    // Pass 1: scan the directory, drop .tmp orphans, classify regular entries
    // (prompt-NNN.kv) vs live-checkpoint files (slot-N.kv). Tracks max_id from
    // existing regular entries so we can give live checkpoints fresh IDs.
    std::vector<std::string> prompt_files;
    std::vector<std::string> slot_files;
    uint64_t max_id = 0;
    bool max_id_found = false;

    for (const auto & entry : fs::directory_iterator(disk_dir, ec)) {
        if (!entry.is_regular_file(ec)) continue;

        const auto path = entry.path();
        const std::string fname = path.filename().string();

        if (path.extension() == ".tmp") {
            std::remove(path.c_str());
            LLAMA_LOG_INFO(" - disk bootstrap: removed orphaned partial file %s\n", fname.c_str());
            continue;
        }

        if (path.extension() != ".kv") continue;

        if (fname.size() == 26 && fname.starts_with("prompt-")) {
            try {
                uint64_t id = std::stoull(fname.substr(7, 16), nullptr, 16);
                if (!max_id_found || id > max_id) {
                    max_id = id;
                    max_id_found = true;
                }
            } catch (...) {
                // not a parseable id; still treat as a regular file below
            }
            prompt_files.push_back(fname);
        } else if (fname.starts_with("slot-")) {
            slot_files.push_back(fname);
        }
        // any other .kv shape is ignored on purpose.
    }

    // Pass 2: promote live-checkpoint files to regular entries by renaming
    // them to prompt-<next_id>.kv. Without this step, the next run of the
    // same slot id would atomically overwrite slot-N.kv via live_checkpoint
    // and the entry loaded in Pass 3 would point at a file whose data_size
    // no longer matches the stored data_disk_size, silently failing restore.
    uint64_t next_id = max_id_found ? max_id + 1 : 0;
    for (const auto & fname : slot_files) {
        char dst_buf[40];
        std::snprintf(dst_buf, sizeof(dst_buf), "prompt-%016llx.kv",
                      (unsigned long long) next_id);
        const std::string src_path = disk_dir + "/" + fname;
        const std::string dst_path = disk_dir + "/" + dst_buf;
        if (std::rename(src_path.c_str(), dst_path.c_str()) == 0) {
            LLAMA_LOG_INFO(" - disk bootstrap: promoted live checkpoint %s -> %s\n",
                           fname.c_str(), dst_buf);
            prompt_files.push_back(dst_buf);
            max_id = next_id;
            max_id_found = true;
            ++next_id;
        } else {
            LLAMA_LOG_INFO(" - disk bootstrap: failed to promote %s, removing\n", fname.c_str());
            std::remove(src_path.c_str());
        }
    }

    int count = 0;

    // Pass 3: read each file's header + token blob and push into states.
    for (const auto & fname : prompt_files) {
        const std::string fpath = disk_dir + "/" + fname;

        FILE * f = std::fopen(fpath.c_str(), "rb");
        if (!f) continue;

        kv_file_header hdr{};
        if (std::fread(&hdr, sizeof(hdr), 1, f) != 1 ||
            std::memcmp(hdr.magic, KV_MAGIC, sizeof(hdr.magic)) != 0 ||
            hdr.version != KV_VERSION) {
            std::fclose(f);
            std::remove(fpath.c_str());
            LLAMA_LOG_INFO(" - disk bootstrap: removed corrupted file %s\n", fname.c_str());
            continue;
        }

        std::vector<uint32_t> tok_buf(hdr.token_count);
        if (hdr.token_count > 0 && std::fread(tok_buf.data(), sizeof(uint32_t), hdr.token_count, f) != hdr.token_count) {
            std::fclose(f);
            std::remove(fpath.c_str());
            LLAMA_LOG_INFO(" - disk bootstrap: removed truncated file %s\n", fname.c_str());
            continue;
        }
        std::fclose(f);

        server_prompt p;
        std::vector<llama_token> t_vec;
        t_vec.reserve(hdr.token_count);
        for (uint32_t t : tok_buf) {
            t_vec.push_back((llama_token)t);
        }
        p.tokens = server_tokens(t_vec, false);
        p.n_kept_prompt = hdr.n_kept_prompt;
        p.n_discarded_prompt = hdr.n_discarded_prompt;
        p.disk_file = fname;
        p.data_disk_size = hdr.data_size;
        p.last_reason    = hdr.save_reason;

        states.push_back(std::move(p));
        count++;
    }

    if (max_id_found) {
        next_disk_id = max_id + 1;
    }

    // Reorder by restore priority: entries that were actively in use or
    // cleanly saved (continued, shutdown) come first so load() is more likely
    // to hit them early; cold/unknown go last so they are also the first to
    // be sacrificed by enforce_disk_limit. Stable sort preserves insertion
    // order (≈ mtime) within each priority class.
    auto priority = [](uint8_t r) -> int {
        using R = disk_save_reason;
        switch ((R) r) {
            case R::continued: return 0;
            case R::shutdown:  return 1;
            case R::evict:     return 2;
            case R::unknown:   return 3;
            case R::cold:      return 4;
        }
        return 3;
    };
    states.sort([&](const server_prompt & a, const server_prompt & b) {
        return priority(a.last_reason) < priority(b.last_reason);
    });

    if (count > 0) {
        LLAMA_LOG_INFO(" - disk bootstrap: loaded %d entries from %s (next id: %llu)\n",
                       count, disk_dir.c_str(), (unsigned long long)next_disk_id);
    }
}
