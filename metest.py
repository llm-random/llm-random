type(model) ----------------------------------------------------
GPTModel(
  (module): Float16Module(
    (module): GPTModel(
      (embedding): LanguageModelEmbedding(
        (word_embeddings): VocabParallelEmbedding()
        (embedding_dropout): Dropout(p=0.0, inplace=False)
      )
      (rotary_pos_emb): RotaryEmbedding()
      (decoder): TransformerBlock(
        (layers): ModuleList(
          (0-15): 16 x TransformerLayer(
            (input_layernorm): LayerNorm()
            (self_attention): SelfAttention(
              (core_attention): TEDotProductAttention(
                (flash_attention): FlashAttention()
                (fused_attention): FusedAttention()
                (unfused_attention): UnfusedDotProductAttention(
                  (scale_mask_softmax): FusedScaleMaskSoftmax()
                  (attention_dropout): Dropout(p=0.0, inplace=False)
                )
              )
              (linear_proj): RowParallelLinear(in_features=1024, out_features=768, bias=False, TP=1)
              (linear_qkv): ColumnParallelLinear(in_features=768, out_features=3072, bias=False, TP=1)
              (q_layernorm): IdentityOp()
              (k_layernorm): IdentityOp()
            )
            (pre_cross_attn_layernorm): IdentityOp()
            (cross_attention): IdentityOp()
            (cross_attn_bda): IdentityFuncOp()
            (pre_mlp_layernorm): LayerNorm()
            (mlp): MLP(
              (linear_fc1): ColumnParallelLinear(in_features=768, out_features=768, bias=False, TP=1)
              (linear_fc2): RowParallelLinear(in_features=768, out_features=768, bias=False, TP=1)
            )
          )
        )
        (final_layernorm): LayerNorm()
      )
      (output_layer): ColumnParallelLinear(in_features=768, out_features=50304, bias=False, TP=1)
    )
  )
)

end model ----------------------------------------------------
model(model) 
GPTModel(
  (module): Float16Module(
    (module): GPTModel(
      (embedding): LanguageModelEmbedding(
        (word_embeddings): VocabParallelEmbedding()
        (embedding_dropout): Dropout(p=0.0, inplace=False)
      )
      (rotary_pos_emb): RotaryEmbedding()
      (decoder): TransformerBlock(
        (layers): ModuleList(
          (0-15): 16 x TransformerLayer(
            (input_layernorm): LayerNorm()
            (self_attention): SelfAttention(
              (core_attention): TEDotProductAttention(
                (flash_attention): FlashAttention()
                (fused_attention): FusedAttention()
                (unfused_attention): UnfusedDotProductAttention(
                  (scale_mask_softmax): FusedScaleMaskSoftmax()
                  (attention_dropout): Dropout(p=0.0, inplace=False)
                )
              )
              (linear_proj): RowParallelLinear(in_features=1024, out_features=768, bias=False, TP=1)
              (linear_qkv): ColumnParallelLinear(in_features=768, out_features=3072, bias=False, TP=1)
              (q_layernorm): IdentityOp()
              (k_layernorm): IdentityOp()
            )
            (pre_cross_attn_layernorm): IdentityOp()
            (cross_attention): IdentityOp()
            (cross_attn_bda): IdentityFuncOp()
            (pre_mlp_layernorm): LayerNorm()
            (mlp): MLP(
              (linear_fc1): ColumnParallelLinear(in_features=768, out_features=768, bias=False, TP=1)
              (linear_fc2): RowParallelLinear(in_features=768, out_features=768, bias=False, TP=1)
            )
          )
        )
        (final_layernorm): LayerNorm()
      )
      (output_layer): ColumnParallelLinear(in_features=768, out_features=50304, bias=False, TP=1)
    )
  )
)

