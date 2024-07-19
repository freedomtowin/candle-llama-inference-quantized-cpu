
#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

// Importing the helper module
pub mod helper;

use anyhow::{bail, Error as E, Result};
use clap::{Parser, ValueEnum};
use std::io::Write;
use tokenizers::Tokenizer;



use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_core::utils::{with_simd128, with_f16c, with_neon, with_avx};
use candle_core::{Device, DType, Tensor, Error, cuda};
use candle_core::quantized::{ggml_file, gguf_file};

use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::utils::{apply_repeat_penalty};
use candle_transformers::models::quantized_llama as model;
use model::ModelWeights;

use hf_hub::api::sync::Api;
use hf_hub::api::sync::ApiRepo;
use hf_hub::{Repo, RepoType};
use helper::{device, hub_load_safetensors};
use helper::{TokenOutputStream};

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum)]

enum Which {
    #[value(name = "7b")]
    L7b,
    #[value(name = "13b")]
    L13b,
    #[value(name = "70b")]
    L70b,
    #[value(name = "7b-chat")]
    L7bChat,
    #[value(name = "13b-chat")]
    L13bChat,
    #[value(name = "70b-chat")]
    L70bChat,
    #[value(name = "7b-code")]
    L7bCode,
    #[value(name = "13b-code")]
    L13bCode,
    #[value(name = "32b-code")]
    L34bCode,
    #[value(name = "7b-leo")]
    Leo7b,
    #[value(name = "13b-leo")]
    Leo13b,
    #[value(name = "7b-mistral")]
    Mistral7b,
    #[value(name = "7b-mistral-instruct")]
    Mistral7bInstruct,
    #[value(name = "7b-mistral-instruct-v0.2")]
    Mistral7bInstructV02,
    #[value(name = "7b-zephyr-a")]
    Zephyr7bAlpha,
    #[value(name = "7b-zephyr-b")]
    Zephyr7bBeta,
    #[value(name = "7b-open-chat-3.5")]
    OpenChat35,
    #[value(name = "7b-starling-a")]
    Starling7bAlpha,
    #[value(name = "mixtral")]
    Mixtral,
    #[value(name = "mixtral-instruct")]
    MixtralInstruct,
    #[value(name = "llama3-8b")]
    L8b,
    #[value(name = "phi3")]
    Phi3,
}

impl Which {
    fn is_mistral(&self) -> bool {
        match self {
            Self::L7b
            | Self::L13b
            | Self::L70b
            | Self::L7bChat
            | Self::L13bChat
            | Self::L70bChat
            | Self::L7bCode
            | Self::L13bCode
            | Self::L34bCode
            | Self::Leo7b
            | Self::Leo13b
            | Self::L8b
            | Self::Phi3 => false,
            // Zephyr and OpenChat are fine tuned versions of mistral and should be treated in the
            // same way. Starling is a fine tuned version of OpenChat.
            Self::OpenChat35
            | Self::Starling7bAlpha
            | Self::Zephyr7bAlpha
            | Self::Zephyr7bBeta
            | Self::Mixtral
            | Self::MixtralInstruct
            | Self::Mistral7b
            | Self::Mistral7bInstruct
            | Self::Mistral7bInstructV02 => true,
        }
    }

    fn is_zephyr(&self) -> bool {
        match self {
            Self::L7b
            | Self::L13b
            | Self::L70b
            | Self::L7bChat
            | Self::L13bChat
            | Self::L70bChat
            | Self::L7bCode
            | Self::L13bCode
            | Self::L34bCode
            | Self::Leo7b
            | Self::Leo13b
            | Self::Mixtral
            | Self::MixtralInstruct
            | Self::Mistral7b
            | Self::Mistral7bInstruct
            | Self::Mistral7bInstructV02
            | Self::OpenChat35
            | Self::Starling7bAlpha
            | Self::L8b
            | Self::Phi3 => false,
            Self::Zephyr7bAlpha | Self::Zephyr7bBeta => true,
        }
    }

    fn is_open_chat(&self) -> bool {
        match self {
            Self::L7b
            | Self::L13b
            | Self::L70b
            | Self::L7bChat
            | Self::L13bChat
            | Self::L70bChat
            | Self::L7bCode
            | Self::L13bCode
            | Self::L34bCode
            | Self::Leo7b
            | Self::Leo13b
            | Self::Mixtral
            | Self::MixtralInstruct
            | Self::Mistral7b
            | Self::Mistral7bInstruct
            | Self::Mistral7bInstructV02
            | Self::Zephyr7bAlpha
            | Self::Zephyr7bBeta
            | Self::L8b
            | Self::Phi3 => false,
            Self::OpenChat35 | Self::Starling7bAlpha => true,
        }
    }

    fn tokenizer_repo(&self) -> &'static str {
        match self {
            Self::L7b
            | Self::L13b
            | Self::L70b
            | Self::L7bChat
            | Self::L13bChat
            | Self::L70bChat
            | Self::L7bCode
            | Self::L13bCode
            | Self::L34bCode => "hf-internal-testing/llama-tokenizer",
            Self::Leo7b => "LeoLM/leo-hessianai-7b",
            Self::Leo13b => "LeoLM/leo-hessianai-13b",
            Self::Mixtral => "mistralai/Mixtral-8x7B-v0.1",
            Self::MixtralInstruct => "mistralai/Mixtral-8x7B-Instruct-v0.1",
            Self::Mistral7b
            | Self::Mistral7bInstruct
            | Self::Mistral7bInstructV02
            | Self::Zephyr7bAlpha
            | Self::Zephyr7bBeta => "mistralai/Mistral-7B-v0.1",
            Self::OpenChat35 => "openchat/openchat_3.5",
            Self::Starling7bAlpha => "berkeley-nest/Starling-LM-7B-alpha",
            Self::L8b => "meta-llama/Meta-Llama-3-8B",
            Self::Phi3 => "microsoft/Phi-3-mini-4k-instruct",
        }
    }
}

// Hardcoded values for L8b model
const EOS_TOKEN: &str = "<|eot_id|>";
const DEFAULT_PROMPT: &str = "What is your favorite theorem?";
const MODEL: &str = "QuantFactory/Meta-Llama-3-8B-GGUF/Meta-Llama-3-8B.Q4_K_S.gguf";
const TOKENIZER: &str = "meta-llama/Meta-Llama-3-8B/tokenizer.json";
const SAMPLE_LEN: usize = 1000;
const TEMPERATURE: f64 = 0.1;
const TOP_P: Option<f64> = None;
const TOP_K: Option<usize> = None;
const SEED: u64 = 299792458;
const TRACING: bool = true;
const VERBOSE_PROMPT: bool = true;
const SPLIT_PROMPT: bool = false;
const CPU: bool = false;
const REPEAT_PENALTY: f32 = 1.1;
const REPEAT_LAST_N: usize = 128;
const WHICH: Which = Which::L8b;
const GQA: Option<usize> = None;
const FORCE_DMMV: bool = false;
const PROMPT: &str = DEFAULT_PROMPT;

fn format_size(size_in_bytes: usize) -> String {
    if size_in_bytes < 1_000 {
        format!("{}B", size_in_bytes)
    } else if size_in_bytes < 1_000_000 {
        format!("{:.2}KB", size_in_bytes as f64 / 1e3)
    } else if size_in_bytes < 1_000_000_000 {
        format!("{:.2}MB", size_in_bytes as f64 / 1e6)
    } else {
        format!("{:.2}GB", size_in_bytes as f64 / 1e9)
    }
}

fn tokenizer() -> anyhow::Result<Tokenizer> {
    let tokenizer_path = match std::path::Path::new(TOKENIZER).exists() {
        true => std::path::PathBuf::from(TOKENIZER),
        false => {
            let api = Api::new()?;
            let repo = WHICH.tokenizer_repo();
            let api = api.repo(Repo::with_revision(repo.to_string(), RepoType::Model, "main".to_string()));
            api.get("tokenizer.json")?
        }
    };
    Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)
}

fn main() -> anyhow::Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    #[cfg(feature = "cuda")]
    candle::quantized::cuda::set_force_dmmv(FORCE_DMMV);

    cuda::set_gemm_reduced_precision_f16(true);
    cuda::set_gemm_reduced_precision_bf16(true);

    let _guard = if TRACING {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };
    
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        with_avx(),
        with_neon(),
        with_simd128(),
        with_f16c()
    );
    println!(
        "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
        TEMPERATURE, REPEAT_PENALTY, REPEAT_LAST_N
    );

    let model_path = match std::path::Path::new(MODEL).exists() {
        true => std::path::PathBuf::from(MODEL),
        false => {
            let api = Api::new()?;
            let repo = "QuantFactory/Meta-Llama-3-8B-GGUF";
            let api = api.repo(Repo::with_revision(repo.to_string(), RepoType::Model, "main".to_string()));
            api.get("Meta-Llama-3-8B.Q4_K_S.gguf")?
        }
    };
    let mut file = std::fs::File::open(&model_path)?;
    let start = std::time::Instant::now();
    let device = device(CPU)?;

    let mut model = match model_path.extension().and_then(|v| v.to_str()) {
        Some("gguf") => {
            let model = gguf_file::Content::read(&mut file).map_err(|e| e.with_path(model_path))?;
            let mut total_size_in_bytes = 0;
            for (_, tensor) in model.tensor_infos.iter() {
                let elem_count = tensor.shape.elem_count();
                total_size_in_bytes +=
                    elem_count * tensor.ggml_dtype.type_size() / tensor.ggml_dtype.block_size();
            }
            println!(
                "loaded {:?} tensors ({}) in {:.2}s",
                model.tensor_infos.len(),
                &format_size(total_size_in_bytes),
                start.elapsed().as_secs_f32(),
            );
            ModelWeights::from_gguf(model, &mut file, &device)?
        }
        Some("ggml" | "bin") | Some(_) | None => {
            let model = ggml_file::Content::read(&mut file, &device)
                .map_err(|e| e.with_path(model_path))?;
            let mut total_size_in_bytes = 0;
            for (_, tensor) in model.tensors.iter() {
                let elem_count = tensor.shape().elem_count();
                total_size_in_bytes +=
                    elem_count * tensor.dtype().type_size() / tensor.dtype().block_size();
            }
            println!(
                "loaded {:?} tensors ({}) in {:.2}s",
                model.tensors.len(),
                &format_size(total_size_in_bytes),
                start.elapsed().as_secs_f32(),
            );
            println!("params: {:?}", model.hparams);
            let default_gqa = match WHICH {
                Which::L7b
                | Which::L13b
                | Which::L7bChat
                | Which::L13bChat
                | Which::L7bCode
                | Which::L13bCode
                | Which::L34bCode
                | Which::Leo7b
                | Which::Leo13b
                | Which::L8b
                | Which::Phi3 => 1,
                Which::Mixtral
                | Which::MixtralInstruct
                | Which::Mistral7b
                | Which::Mistral7bInstruct
                | Which::Mistral7bInstructV02
                | Which::Zephyr7bAlpha
                | Which::Zephyr7bBeta
                | Which::L70b
                | Which::L70bChat
                | Which::OpenChat35
                | Which::Starling7bAlpha => 8,
            };
            ModelWeights::from_ggml(model, GQA.unwrap_or(default_gqa))?
        }
    };
    println!("model built");

    let tokenizer = tokenizer()?;
    let mut tos = TokenOutputStream::new(tokenizer);
    let prompt = PROMPT.to_string();


    let eos_token = match tos.tokenizer().get_vocab(true).get(EOS_TOKEN) {
        Some(&token) => token,
        None => {
            eprintln!("EOS token 1 not found in the tokenizer vocabulary.");
            return Ok(());
        }
    };

    let mut pre_prompt_tokens = vec![];
    for prompt_index in 0.. {
        let prompt_str = prompt.clone();
        print!("{}", &prompt_str);
        let tokens = tos
            .tokenizer()
            .encode(prompt_str, true)
            .map_err(anyhow::Error::msg)?;
        if VERBOSE_PROMPT {
            for (token, id) in tokens.get_tokens().iter().zip(tokens.get_ids().iter()) {
                let token = token.replace('▁', " ").replace("<0x0A>", "\n");
                println!("{id:7} -> '{token}'");
            }
        }

        let prompt_tokens = [&pre_prompt_tokens, tokens.get_ids()].concat();
        let to_sample = SAMPLE_LEN.saturating_sub(1);
        let prompt_tokens = if prompt_tokens.len() + to_sample > model::MAX_SEQ_LEN - 10 {
            let to_remove = prompt_tokens.len() + to_sample + 10 - model::MAX_SEQ_LEN;
            prompt_tokens[prompt_tokens.len().saturating_sub(to_remove)..].to_vec()
        } else {
            prompt_tokens
        };
        let mut all_tokens = vec![];
        let mut logits_processor = {
            let temperature = TEMPERATURE;
            let sampling = if temperature <= 0. {
                Sampling::ArgMax
            } else {
                match (TOP_K, TOP_P) {
                    (None, None) => Sampling::All { temperature },
                    (Some(k), None) => Sampling::TopK { k, temperature },
                    (None, Some(p)) => Sampling::TopP { p, temperature },
                    (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
                }
            };
            LogitsProcessor::from_sampling(SEED, sampling)
        };

        let start_prompt_processing = std::time::Instant::now();
        let mut next_token = if !SPLIT_PROMPT {
            let input = Tensor::new(prompt_tokens.as_slice(), &device)?.unsqueeze(0)?;
            let logits = model.forward(&input, 0)?;
            let logits = logits.squeeze(0)?;
            logits_processor.sample(&logits)?
        } else {
            let mut next_token = 0;
            for (pos, token) in prompt_tokens.iter().enumerate() {
                let input = Tensor::new(&[*token], &device)?.unsqueeze(0)?;
                let logits = model.forward(&input, pos)?;
                let logits = logits.squeeze(0)?;
                next_token = logits_processor.sample(&logits)?
            }
            next_token
        };
        let prompt_dt = start_prompt_processing.elapsed();
        all_tokens.push(next_token);
        if let Some(t) = tos.next_token(next_token)? {
            print!("{t}");
            std::io::stdout().flush()?;
        }

        let start_post_prompt = std::time::Instant::now();
        let mut sampled = 0;
        for index in 0..to_sample {
            let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
            let logits = model.forward(&input, prompt_tokens.len() + index)?;
            let logits = logits.squeeze(0)?;
            let logits = if REPEAT_PENALTY == 1. {
                logits
            } else {
                let start_at = all_tokens.len().saturating_sub(REPEAT_LAST_N);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    REPEAT_PENALTY,
                    &all_tokens[start_at..],
                )?
            };
            next_token = logits_processor.sample(&logits)?;
            all_tokens.push(next_token);
            if let Some(t) = tos.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
            sampled += 1;
            if next_token == eos_token {
                break;
            };
        }
        if let Some(rest) = tos.decode_rest().map_err(E::msg)? {
            print!("{rest}");
        }
        std::io::stdout().flush()?;
        let dt = start_post_prompt.elapsed();
        println!(
            "\n\n{:4} prompt tokens processed: {:.2} token/s",
            prompt_tokens.len(),
            prompt_tokens.len() as f64 / prompt_dt.as_secs_f64(),
        );
        println!(
            "{sampled:4} tokens generated: {:.2} token/s",
            sampled as f64 / dt.as_secs_f64(),
        );

        break; // Since we're using a single prompt, break after the first iteration
    }

    Ok(())
}