use std::fmt::format;

use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use bytemuck::cast_slice;
use safetensors::SafeTensors;
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        println!("{:?}", config);
        safetensor.iter().for_each(|it| println!("name = {:?}, shape = {:?}", it.0, it.1.shape()));

        assert_eq!(config.torch_dtype, "float32");
        assert!(config.num_hidden_layers > 0);

        let embedding_table = safetensor.tensor("lm_head.weight").unwrap();
        assert_eq!(embedding_table.shape().len(), 2);
        assert_eq!(embedding_table.shape()[0], config.vocab_size);
        let embedding_table = Tensor::new(cast_slice(embedding_table.data()).to_vec(), &embedding_table.shape().to_vec());

        // decoder layer
        let mut rms_att_w = Vec::new();
        let mut wq = Vec::new();
        let mut wk = Vec::new();
        let mut wv = Vec::new();
        let mut wo = Vec::new();

        // ffn layer
        let mut rms_ffn_w = Vec::new();
        let mut w_up = Vec::new();
        let mut w_gate = Vec::new();
        let mut w_down = Vec::new();

        // output
        let rms_out_w = safetensor.tensor("model.norm.weight").unwrap();
        assert_eq!(rms_out_w.shape().len(), 1);
        assert_eq!(rms_out_w.shape()[0], config.hidden_size);
        let rms_out_w = Tensor::new(cast_slice(rms_out_w.data()).to_vec(), &rms_out_w.shape().to_vec());

        for i in 0..config.num_hidden_layers {
            // decoder layer
            rms_att_w.push(safetensor.tensor(&format!("model.layers.{}.input_layernorm.weight", i)).unwrap());
            wq.push(safetensor.tensor(&format!("model.layers.{}.self_attn.q_proj.weight", i)).unwrap());
            wk.push(safetensor.tensor(&format!("model.layers.{}.self_attn.k_proj.weight", i)).unwrap());
            wv.push(safetensor.tensor(&format!("model.layers.{}.self_attn.v_proj.weight", i)).unwrap());
            wo.push(safetensor.tensor(&format!("model.layers.{}.self_attn.o_proj.weight", i)).unwrap());
            
            // ffn layer
            rms_ffn_w.push(safetensor.tensor(&format!("model.layers.{}.post_attention_layernorm.weight", i)).unwrap());
            w_up.push(safetensor.tensor(&format!("model.layers.{}.mlp.up_proj.weight", i)).unwrap());
            w_gate.push(safetensor.tensor(&format!("model.layers.{}.mlp.gate_proj.weight", i)).unwrap());
            w_down.push(safetensor.tensor(&format!("model.layers.{}.mlp.down_proj.weight", i)).unwrap());
        }

        let rms_att_w = rms_att_w.iter().map(|t| {
            assert_eq!(t.shape().len(), 1);
            assert_eq!(t.shape()[0], config.hidden_size);
            Tensor::new(cast_slice(t.data()).to_vec(), &t.shape().to_vec())
        }).collect();

        assert!(wq[0].shape().len() == 2 && wq[0].shape()[0] % config.num_attention_heads == 0);
        let qkv_vec_dim = wq[0].shape()[0] / config.num_attention_heads;

        let wq = wq.iter().map(|t| {
            assert_eq!(t.shape().len(), 2);
            assert_eq!(t.shape()[0], config.num_attention_heads * qkv_vec_dim);
            Tensor::new(cast_slice(t.data()).to_vec(), &t.shape().to_vec())
        }).collect();
        let wk = wk.iter().map(|t| {
            assert_eq!(t.shape().len(), 2);
            assert_eq!(t.shape()[0], config.num_key_value_heads * qkv_vec_dim);
            Tensor::new(cast_slice(t.data()).to_vec(), &t.shape().to_vec())
        }).collect();
        let wv = wv.iter().map(|t| {
            assert_eq!(t.shape().len(), 2);
            assert_eq!(t.shape()[0], config.num_key_value_heads * qkv_vec_dim);
            Tensor::new(cast_slice(t.data()).to_vec(), &t.shape().to_vec())
        }).collect();
        let wo = wo.iter().map(|t| {
            assert_eq!(t.shape().len(), 2);
            assert_eq!(t.shape()[0], config.hidden_size);
            assert_eq!(t.shape()[1], qkv_vec_dim * config.num_attention_heads);
            Tensor::new(cast_slice(t.data()).to_vec(), &t.shape().to_vec())
        }).collect();


        let rms_ffn_w = rms_ffn_w.iter().map(|t| {
            assert_eq!(t.shape().len(), 1);
            assert_eq!(t.shape()[0], config.hidden_size);
            Tensor::new(cast_slice(t.data()).to_vec(), &t.shape().to_vec())
        }).collect();
        let w_gate = w_gate.iter().map(|t| {
            assert_eq!(t.shape().len(), 2);
            assert_eq!(t.shape()[1], config.hidden_size);
            assert_eq!(t.shape()[0], config.intermediate_size);
            Tensor::new(cast_slice(t.data()).to_vec(), &t.shape().to_vec())
        }).collect();
        let w_up = w_up.iter().map(|t| {
            assert_eq!(t.shape().len(), 2);
            assert_eq!(t.shape()[1], config.hidden_size);
            assert_eq!(t.shape()[0], config.intermediate_size);
            Tensor::new(cast_slice(t.data()).to_vec(), &t.shape().to_vec())
        }).collect();
        let w_down = w_down.iter().map(|t| {
            assert_eq!(t.shape().len(), 2);
            assert_eq!(t.shape()[0], config.hidden_size);
            assert_eq!(t.shape()[1], config.intermediate_size);
            Tensor::new(cast_slice(t.data()).to_vec(), &t.shape().to_vec())
        }).collect();

        let lm_head = safetensor.tensor("lm_head.weight").unwrap();
        let lm_head = Tensor::new(cast_slice(lm_head.data()).to_vec(), &lm_head.shape().to_vec());

        LLamaParams {
            embedding_table,
            rms_att_w,
            wq,
            wk,
            wv,
            wo,
            rms_ffn_w,
            w_up,
            w_gate,
            w_down,
            rms_out_w,
            lm_head,
        }
    }
}
