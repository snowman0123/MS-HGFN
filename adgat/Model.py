from Layers import *



class AD_GAT(nn.Module):
    def __init__(self, num_stock, d_market, hidn_rnn, heads_att, hidn_att, dropout=0, alpha=0.2, t_mix = 1, infer = 1, relation_static = 0):
        super(AD_GAT, self).__init__()
        self.t_mix = t_mix
        self.dropout = dropout

        self.GRUs_s = Graph_GRUModel(num_stock, d_market, hidn_rnn)
        self.GRUs_r = Graph_GRUModel(num_stock, d_market, hidn_rnn)

        self.attentions = [
            Graph_Attention(hidn_rnn, hidn_att, dropout=dropout, alpha=alpha, residual=True, concat=True) for _
            in range(heads_att)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.X2Os = Graph_Linear(num_stock, heads_att * hidn_att  + hidn_rnn , 2, bias = True)
        self.reset_parameters()

    def reset_parameters(self):
        reset_parameters(self.named_parameters)

    def get_relation(self,x_numerical, x_textual, relation_static = None):
        x_r = self.tensor(x_numerical, x_textual)
        x_r = self.GRUs_r(x_r)
        relation = torch.stack([att.get_relation(x_r, relation_static=relation_static) for att in self.attentions])
        # relation_mean = torch.mean(abs_relation,dim = 1)
        return relation

    def get_gate(self,x_numerical,x_textual):
        x_s = self.tensor(x_numerical, x_textual)
        x_s = self.GRUs_s(x_s)
        gate = torch.stack([att.get_gate(x_s) for att in self.attentions])
        return gate

    def forward(self, x_market, relation_static = None):
        x_r = self.GRUs_r(x_market)
        x_s = self.GRUs_s(x_market)

        x_r = F.dropout(x_r, self.dropout, training=self.training)
        x_s = F.dropout(x_s, self.dropout, training=self.training)

        x = torch.cat([att(x_s, x_r, relation_static = relation_static) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([x, x_s], dim=1)
        x = F.elu(self.X2Os(x))


        output = F.log_softmax(x, dim=1)
        return output

