class EmRelModel(BertPreTrainedModel):
    def __init__(self, config, num_labels, max_ent_cnt, with_naive_feature=False, entity_structure=False):
        super().__init__(config)
        self.num_labels = num_labels
        self.max_ent_cnt = max_ent_cnt
        self.with_naive_feature = with_naive_feature
        self.feature_size = self.cls_size = 128

        self.roberta = RobertaModel(config, with_naive_feature, entity_structure)
        self.head_fc = nn.Linear(config.hidden_size, self.feature_size)
        self.tail_fc = nn.Linear(config.hidden_size, self.feature_size)

        if self.with_naive_feature:
            self.cls_size += 20
            self.distance_emb = nn.Embedding(20, 20, padding_idx=10)
        # Tucker Decomp classifier
        self.proto_dim = 128
        self.core_tensor = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(self.cls_size, self.proto_dim, self.cls_size)))
        self.proto_repre = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_labels, self.proto_dim)))
        self.cls_bias = nn.Parameter(torch.zeros(num_labels))

        # Cross Attention
        self.head2proto_att = MultiheadAttention(self.proto_dim, self.feature_size, self.feature_size, self.feature_size, 4, config.layer_norm_eps)
        self.tail2proto_att = MultiheadAttention(self.proto_dim, self.feature_size, self.feature_size, self.feature_size, 4, config.layer_norm_eps)
        self.context2proto_att = MultiheadAttention(self.proto_dim, config.hidden_size, config.hidden_size, config.hidden_size, 4, config.layer_norm_eps)
        self.proto2headent_att = MultiheadAttention(self.feature_size, self.proto_dim, self.proto_dim, self.feature_size, 4, config.layer_norm_eps)
        self.proto2tailent_att = MultiheadAttention(self.feature_size, self.proto_dim, self.proto_dim, self.feature_size, 4, config.layer_norm_eps)
        self.post_layernorm = torch.nn.LayerNorm(self.proto_dim, config.layer_norm_eps)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            ent_mask=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            ent_ner=None,
            ent_pos=None,
            ent_distance=None,
            label=None,
            label_mask=None,
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            ner_ids=ent_ner,
            ent_ids=ent_pos,
        )
        # get sequence outputs
        outputs = outputs[0]
        ent = torch.matmul(ent_mask, outputs)
        ent_head = self.head_fc(ent)
        ent_tail = self.tail_fc(ent)
        bs = outputs.shape[0]

        # # ===== w/o Fusion
        # ent_head = ent_head.unsqueeze(2).repeat(1, 1, self.max_ent_cnt, 1)
        # ent_tail = ent_tail.unsqueeze(1).repeat(1, self.max_ent_cnt, 1, 1)
        # if self.with_naive_feature:
        #     ent_head = torch.cat([ent_head, self.distance_emb(ent_distance)], dim=-1)
        #     ent_tail = torch.cat([ent_tail, self.distance_emb(20 - ent_distance)], dim=-1)
        # # Align
        # logits = torch.einsum("xyz,bhtx,bcy,bhtz->bhtc", self.core_tensor, ent_head, self.proto_repre.unsqueeze(0).expand([bs, -1, -1]), ent_tail) + self.cls_bias

        # ===== w/ Fusion
        ent2proto_mask = torch.sum(ent_mask, dim=2).unsqueeze(1).repeat(1, self.num_labels, 1)
        proto_headaware, _ = self.head2proto_att(self.proto_repre.unsqueeze(0), ent_head, ent_head, ent2proto_mask)
        proto_tailaware, _ = self.tail2proto_att(self.proto_repre.unsqueeze(0), ent_tail, ent_tail, ent2proto_mask)
        proto_contextaware, _ = self.context2proto_att(self.proto_repre.unsqueeze(0), outputs, outputs, attention_mask.unsqueeze(1).repeat(1, self.num_labels, 1))
        ent_head, _ = self.proto2headent_att(ent_head, self.proto_repre.unsqueeze(0), self.proto_repre.unsqueeze(0))
        ent_tail, _ = self.proto2tailent_att(ent_tail, self.proto_repre.unsqueeze(0), self.proto_repre.unsqueeze(0))
        proto = self.post_layernorm(proto_headaware + proto_tailaware)
        ent_head = ent_head.unsqueeze(2).repeat(1, 1, self.max_ent_cnt, 1)
        ent_tail = ent_tail.unsqueeze(1).repeat(1, self.max_ent_cnt, 1, 1)
        if self.with_naive_feature:
            ent_head = torch.cat([ent_head, self.distance_emb(ent_distance)], dim=-1)
            ent_tail = torch.cat([ent_tail, self.distance_emb(20 - ent_distance)], dim=-1)
        # ===== Align
        logits = torch.einsum("xyz,bhtx,bcy,bhtz->bhtc", self.core_tensor, ent_head, proto.float(), ent_tail) + self.cls_bias

        loss_fct = BCEWithLogitsLoss(reduction='none')
        loss_all_ent_pair = loss_fct(logits.view(-1, self.num_labels), label.float().view(-1, self.num_labels))
        # loss_all_ent_pair: [bs, max_ent_cnt, max_ent_cnt]
        # label_mask: [bs, max_ent_cnt, max_ent_cnt]
        loss_all_ent_pair = loss_all_ent_pair.view(-1, self.max_ent_cnt, self.max_ent_cnt, self.num_labels)
        loss_all_ent_pair = torch.mean(loss_all_ent_pair, dim=-1)
        loss_per_example = torch.sum(loss_all_ent_pair * label_mask, dim=[1, 2]) / torch.sum(label_mask, dim=[1, 2])
        loss = torch.mean(loss_per_example)

        logits = torch.sigmoid(logits)
        return (loss, logits)  # (loss), logits
