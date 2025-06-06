
import re
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
from torch.nn.utils.rnn import pad_sequence
COMMA = u'\u060C'
SEMICOLON = u'\u061B'
QUESTION = u'\u061F'
HAMZA = u'\u0621'
ALEF_MADDA = u'\u0622'
ALEF_HAMZA_ABOVE = u'\u0623'
WAW_HAMZA = u'\u0624'
ALEF_HAMZA_BELOW = u'\u0625'
YEH_HAMZA = u'\u0626'
ALEF = u'\u0627'
BEH = u'\u0628'
TEH_MARBUTA = u'\u0629'
TEH = u'\u062a'
THEH = u'\u062b'
JEEM = u'\u062c'
HAH = u'\u062d'
KHAH = u'\u062e'
DAL = u'\u062f'
THAL = u'\u0630'
REH = u'\u0631'
ZAIN = u'\u0632'
SEEN = u'\u0633'
SHEEN = u'\u0634'
SAD = u'\u0635'
DAD = u'\u0636'
TAH = u'\u0637'
ZAH = u'\u0638'
AIN = u'\u0639'
GHAIN = u'\u063a'
TATWEEL = u'\u0640'
FEH = u'\u0641'
QAF = u'\u0642'
KAF = u'\u0643'
LAM = u'\u0644'
MEEM = u'\u0645'
NOON = u'\u0646'
HEH = u'\u0647'
WAW = u'\u0648'
ALEF_MAKSURA = u'\u0649'
YEH = u'\u064a'
MADDA_ABOVE = u'\u0653'
HAMZA_ABOVE = u'\u0654'
HAMZA_BELOW = u'\u0655'
ZERO = u'\u0660'
ONE = u'\u0661'
TWO = u'\u0662'
THREE = u'\u0663'
FOUR = u'\u0664'
FIVE = u'\u0665'
SIX = u'\u0666'
SEVEN = u'\u0667'
EIGHT = u'\u0668'
NINE = u'\u0669'
PERCENT = u'\u066a'
DECIMAL = u'\u066b'
THOUSANDS = u'\u066c'
STAR = u'\u066d'
MINI_ALEF = u'\u0670'
ALEF_WASLA = u'\u0671'
FULL_STOP = u'\u06d4'
BYTE_ORDER_MARK = u'\ufeff'

# Diacritics
FATHATAN = u'\u064b'
DAMMATAN = u'\u064c'
KASRATAN = u'\u064d'
FATHA = u'\u064e'
DAMMA = u'\u064f'
KASRA = u'\u0650'
SHADDA = u'\u0651'
SUKUN = u'\u0652'

#Ligatures
LAM_ALEF = u'\ufefb'
LAM_ALEF_HAMZA_ABOVE = u'\ufef7'
LAM_ALEF_HAMZA_BELOW = u'\ufef9'
LAM_ALEF_MADDA_ABOVE = u'\ufef5'
SIMPLE_LAM_ALEF = u'\u0644\u0627'
SIMPLE_LAM_ALEF_HAMZA_ABOVE = u'\u0644\u0623'
SIMPLE_LAM_ALEF_HAMZA_BELOW = u'\u0644\u0625'
SIMPLE_LAM_ALEF_MADDA_ABOVE = u'\u0644\u0622'


HARAKAT_PAT = re.compile(u"["+u"".join([FATHATAN, DAMMATAN, KASRATAN,
                                        FATHA, DAMMA, KASRA, SUKUN,
                                        SHADDA])+u"]")
HAMZAT_PAT = re.compile(u"["+u"".join([WAW_HAMZA, YEH_HAMZA])+u"]")
ALEFAT_PAT = re.compile(u"["+u"".join([ALEF_MADDA, ALEF_HAMZA_ABOVE,
                                       ALEF_HAMZA_BELOW, HAMZA_ABOVE,
                                       HAMZA_BELOW])+u"]")
LAMALEFAT_PAT = re.compile(u"["+u"".join([LAM_ALEF,
                                          LAM_ALEF_HAMZA_ABOVE,
                                          LAM_ALEF_HAMZA_BELOW,
LAM_ALEF_MADDA_ABOVE])+u"]")

def strip_tashkeel(text):
    text = HARAKAT_PAT.sub('', text)
    text = re.sub(u"[\u064E]", "", text,  flags=re.UNICODE) # fattha
    text = re.sub(u"[\u0671]", "", text,  flags=re.UNICODE) # waSla
    return text

def strip_tatweel(text):
    return re.sub(u'[%s]' % TATWEEL, '', text)

# remove removing Tashkeel + removing Tatweel + non Arabic chars
def remove_non_arabic(text):
    text = strip_tashkeel(text)
    text = strip_tatweel(text)
    return ' '.join(re.sub(u"[^\u0621-\u063A\u0641-\u064A ]", " ", text,  flags=re.UNICODE).split())



import sys
import re


buck2uni = {
            "'": u"\u0621", # hamza-on-the-linea
            "|": u"\u0622", # madda
            ">": u"\u0623", # hamza-on-'alif
            "&": u"\u0624", # hamza-on-waaw
            "<": u"\u0625", # hamza-under-'alif
            "}": u"\u0626", # hamza-on-yaa'
            "A": u"\u0627", # bare 'alif
            "b": u"\u0628", # baa'
            "p": u"\u0629", # taa' marbuuTa
            "t": u"\u062A", # taa'
            "v": u"\u062B", # thaa'
            "j": u"\u062C", # jiim
            "H": u"\u062D", # Haa'
            "x": u"\u062E", # khaa'
            "d": u"\u062F", # daal
            "*": u"\u0630", # dhaal
            "r": u"\u0631", # raa'
            "z": u"\u0632", # zaay
            "s": u"\u0633", # siin
            "$": u"\u0634", # shiin
            "S": u"\u0635", # Saad
            "D": u"\u0636", # Daad
            "T": u"\u0637", # Taa'
            "Z": u"\u0638", # Zaa' (DHaa')
            "E": u"\u0639", # cayn
            "g": u"\u063A", # ghayn
            "_": u"\u0640", # taTwiil
            "f": u"\u0641", # faa'
            "q": u"\u0642", # qaaf
            "k": u"\u0643", # kaaf
            "l": u"\u0644", # laam
            "m": u"\u0645", # miim
            "n": u"\u0646", # nuun
            "h": u"\u0647", # haa'
            "w": u"\u0648", # waaw
            "Y": u"\u0649", # 'alif maqSuura
            "y": u"\u064A", # yaa'
            "F": u"\u064B", # fatHatayn
            "N": u"\u064C", # Dammatayn
            "K": u"\u064D", # kasratayn
            "a": u"\u064E", # fatHa
            "u": u"\u064F", # Damma
            "i": u"\u0650", # kasra
            "~": u"\u0651", # shaddah
            "o": u"\u0652", # sukuun
            "`": u"\u0670", # dagger 'alif
            "{": u"\u0671", # waSla
}

# For a reverse transliteration (Unicode -> Buckwalter), a dictionary
# which is the reverse of the above buck2uni is essential.
uni2buck = {}

# Iterate through all the items in the buck2uni dict.
for (key, value) in buck2uni.items():
    # The value from buck2uni becomes a key in uni2buck, and vice
    # versa for the keys.
    uni2buck[value] = key

# add special characters
uni2buck[u"\ufefb"] = "lA"
uni2buck[u"\ufef7"] = "l>"
uni2buck[u"\ufef5"] = "l|"
uni2buck[u"\ufef9"] = "l<"

# clean the arabic text from unwanted characters that may cause problem while building the language model
def clean_text(text):
    text = re.sub(u"[\ufeff]", "", text,  flags=re.UNICODE) # strip Unicode Character 'ZERO WIDTH NO-BREAK SPACE' (U+FEFF). For more info, check http://www.fileformat.info/info/unicode/char/feff/index.htm
    text = remove_non_arabic(text)
    text = strip_tashkeel(text)
    text = strip_tatweel(text)
    return text

# convert a single word into buckwalter and vice versa
def transliterate_word(input_word, direction='bw2ar'):
    output_word = ''
    
    for char in input_word:
        
        if direction == 'bw2ar':
            
            output_word += buck2uni.get(char, char)
        elif direction == 'ar2bw':
            
            output_word += uni2buck.get(char, char)
        else:
            sys.stderr.write('Error: invalid direction!')
            sys.exit()
    return output_word


# convert a text into buckwalter and vice versa
def transliterate_text(input_text, direction='bw2ar'):
    output_text = ''
    for input_word in input_text.split(' '):
        output_text += transliterate_word(input_word, direction) + ' '

    return output_text[:-1] # remove the last space ONLY


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.stderr.write('Usage: INPUT TEXT | python {} DIRECTION(bw2ar|ar2bw)'.format(sys.argv[1]))
        exit(1)
    for line in sys.stdin:
        line = line if sys.argv[1] == 'bw2ar' else clean_text(line)
        output_text = transliterate_text(line, direction=str(sys.argv[1]))
        if output_text.strip() != '':
            sys.stdout.write('{}\n'.format(output_text.strip()))




import re

import torch


# Diacritics
FATHATAN = u'\u064b'
DAMMATAN = u'\u064c'
KASRATAN = u'\u064d'
FATHA = u'\u064e'
DAMMA = u'\u064f'
KASRA = u'\u0650'
SHADDA = u'\u0651'
SUKUN = u'\u0652'
TATWEEL = u'\u0640'

HARAKAT_PAT = re.compile(u"["+u"".join([FATHATAN, DAMMATAN, KASRATAN,
                                        FATHA, DAMMA, KASRA, SUKUN,
                                        SHADDA])+u"]")


class TashkeelTokenizer:

    def __init__(self):
        self.letters = [' ', '$', '&', "'", '*', '<', '>', 'A', 'D', 'E', 'H', 'S', 'T', 'Y', 'Z',
                        'b', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't',
                        'v', 'w', 'x', 'y', 'z', '|', '}'
                       ]
        self.letters = ['<PAD>', '<BOS>', '<EOS>'] + self.letters + ['<MASK>']

        self.no_tashkeel_tag = '<NT>'
        self.tashkeel_list = ['<NT>', '<SD>', '<SDD>', '<SF>', '<SFF>', '<SK>',
                               '<SKK>', 'F', 'K', 'N', 'a', 'i', 'o', 'u', '~']

        self.tashkeel_list = ['<PAD>', '<BOS>', '<EOS>'] + self.tashkeel_list

        self.tashkeel_map = {c:i for i,c in enumerate(self.tashkeel_list)}
        self.letters_map = {c:i for i,c in enumerate(self.letters)}
        self.inverse_tags = {
                 '~a': '<SF>',  # shaddah and fatHa
                 '~u': '<SD>',  # shaddah and Damma
                 '~i': '<SK>',  # shaddah and kasra
                 '~F': '<SFF>', # shaddah and fatHatayn
                 '~N': '<SDD>', # shaddah and Dammatayn
                 '~K': '<SKK>'  # shaddah and kasratayn
        }
        self.tags = {v:k for k,v in self.inverse_tags.items()}
        self.shaddah_last  = ['a~', 'u~', 'i~', 'F~', 'N~', 'K~']
        self.shaddah_first = ['~a', '~u', '~i', '~F', '~N', '~K']
        self.tahkeel_chars = ['F','N','K','a', 'u', 'i', '~', 'o']


    def clean_text(self, text):
        text = re.sub(u'[%s]' % u'\u0640', '', text) # strip tatweel
        text = text.replace('ٱ', 'ا')
        return ' '.join(re.sub(u"[^\u0621-\u063A\u0640-\u0652\u0670\u0671\ufefb\ufef7\ufef5\ufef9 ]", " ", text,  flags=re.UNICODE).split())


    def check_match(self, text_with_tashkeel, letter_n_tashkeel_pairs):
        text_with_tashkeel = text_with_tashkeel.strip()
        # test if the reconstructed text with tashkeel is the same as the original one
        syn_text = self.combine_tashkeel_with_text(letter_n_tashkeel_pairs)
        return syn_text == text_with_tashkeel or syn_text == self.unify_shaddah_position(text_with_tashkeel)


    def unify_shaddah_position(self, text_with_tashkeel):
        # unify the order of shaddah and the harakah to make shaddah always at the beginning
        for i in range(len(self.shaddah_first)):
            text_with_tashkeel = text_with_tashkeel.replace(self.shaddah_last[i], self.shaddah_first[i])
        return text_with_tashkeel


    def split_tashkeel_from_text(self, text_with_tashkeel, test_match=True):
        text_with_tashkeel = self.clean_text(text_with_tashkeel)
        text_with_tashkeel = transliterate_text(text_with_tashkeel, 'ar2bw')
        text_with_tashkeel = text_with_tashkeel.replace('`', '') # remove dagger 'alif

        # unify the order of shaddah and the harakah to make shaddah always at the beginning
        text_with_tashkeel = self.unify_shaddah_position(text_with_tashkeel)

        # remove duplicated harakat
        for i in range(len(self.tahkeel_chars)):
            text_with_tashkeel = text_with_tashkeel.replace(self.tahkeel_chars[i]*2, self.tahkeel_chars[i])

        letter_n_tashkeel_pairs = []
        for i in range(len(text_with_tashkeel)):

            if i < (len(text_with_tashkeel) - 1) and not text_with_tashkeel[i] in self.tashkeel_list and text_with_tashkeel[i+1] in self.tashkeel_list:

                if text_with_tashkeel[i+1] == '~':

                    if i+2 < len(text_with_tashkeel) and f'~{text_with_tashkeel[i+2]}' in self.inverse_tags:
                        letter_n_tashkeel_pairs.append((text_with_tashkeel[i], self.inverse_tags[f'~{text_with_tashkeel[i+2]}']))
                    else:

                        letter_n_tashkeel_pairs.append((text_with_tashkeel[i], '~'))
                else:
                    letter_n_tashkeel_pairs.append((text_with_tashkeel[i], text_with_tashkeel[i+1]))

            elif not text_with_tashkeel[i] in self.tashkeel_list:
                letter_n_tashkeel_pairs.append((text_with_tashkeel[i], self.no_tashkeel_tag))

        if test_match:

            assert self.check_match(text_with_tashkeel, letter_n_tashkeel_pairs)
        return [('<BOS>', '<BOS>')] + letter_n_tashkeel_pairs + [('<EOS>', '<EOS>')]


    def combine_tashkeel_with_text(self, letter_n_tashkeel_pairs):
        combined_with_tashkeel = []
        for letter, tashkeel in letter_n_tashkeel_pairs:
            combined_with_tashkeel.append(letter)
            if tashkeel in self.tags:
                combined_with_tashkeel.append(self.tags[tashkeel])
            elif tashkeel != self.no_tashkeel_tag:
                combined_with_tashkeel.append(tashkeel)
        text = ''.join(combined_with_tashkeel)
        return text


    def encode(self, text_with_tashkeel, test_match=True):
        letter_n_tashkeel_pairs = self.split_tashkeel_from_text(text_with_tashkeel, test_match)
        text, tashkeel = zip(*letter_n_tashkeel_pairs)
        input_ids = [self.letters_map[c] for c in text]
        target_ids = [self.tashkeel_map[c] for c in tashkeel]
        return torch.LongTensor(input_ids), torch.LongTensor(target_ids)


    def filter_tashkeel(self, tashkeel):
        tmp = []
        for i, t in enumerate(tashkeel):
            if i != 0 and t == '<BOS>':
                t = self.no_tashkeel_tag
            elif i != (len(tashkeel) - 1) and t == '<EOS>':
                t = self.no_tashkeel_tag
            tmp.append(t)
        tashkeel = tmp
        return tashkeel


    def decode(self, input_ids, target_ids):

        input_ids = input_ids.cpu().tolist()
        target_ids = target_ids.cpu().tolist()
        ar_texts = []
        for j in range(len(input_ids)):
            letters = [self.letters[i] for i in input_ids[j]]
            tashkeel = [self.tashkeel_list[i] for i in target_ids[j]]

            letters = list(filter(lambda x: x != '<BOS>' and x != '<EOS>' and x != '<PAD>', letters))
            tashkeel = self.filter_tashkeel(tashkeel)
            tashkeel = list(filter(lambda x: x != '<BOS>' and x != '<EOS>' and x != '<PAD>', tashkeel))


            letter_n_tashkeel_pairs = list(zip(letters, tashkeel))
            bw_text = self.combine_tashkeel_with_text(letter_n_tashkeel_pairs)
            ar_text = transliterate_text(bw_text, 'bw2ar')
            ar_texts.append(ar_text)
        return ar_texts

    def get_tashkeel_with_case_ending(self, text, case_ending=True):
        text_split = self.split_tashkeel_from_text(text, test_match=False)
        text_spaces_indecies = [i for i, el in enumerate(text_split) if el == (' ', '<NT>')]
        new_text_split = []
        for i, el in enumerate(text_split):
            if not case_ending and (i+1) in text_spaces_indecies:
                el = (el[0], '<NT>')
            new_text_split.append(el)
        letters, tashkeel = zip(*new_text_split)
        return letters, tashkeel


    def remove_tashkeel(self, text):
        text = HARAKAT_PAT.sub('', text)
        text = re.sub(u"[\u064E]", "", text,  flags=re.UNICODE) # fattha
        text = re.sub(u"[\u0671]", "", text,  flags=re.UNICODE) # waSla
        return text
    

###TRansformersss#########




import math
import torch
import torch.nn as nn


class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, s_mask):
        # 1. compute self attention
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=s_mask)

        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)

        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x


class DecoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = LayerNorm(d_model=d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, dec, enc, t_mask, s_mask):
        # 1. compute self attention
        _x = dec
        x = self.self_attention(q=dec, k=dec, v=dec, mask=t_mask)

        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        if enc is not None:
            # 3. compute encoder - decoder attention
            _x = x
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=s_mask)

            # 4. add and norm
            x = self.dropout2(x)
            x = self.norm2(x + _x)

        # 5. positionwise feed forward network
        _x = x
        x = self.ffn(x)

        # 6. add and norm
        x = self.dropout3(x)
        x = self.norm3(x + _x)
        return x


class ScaleDotProductAttention(nn.Module):


    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):

        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v, score


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_concat = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        # 5. visualize attention map
        # TODO : we should implement visualization

        return out

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):

        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1' means last dimension.

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out


class TransformerEmbedding(nn.Module):
    """
    token embedding + positional encoding (sinusoid)
    positional encoding can give positional information to network
    """

    def __init__(self, vocab_size, d_model, max_len, drop_prob, padding_idx, learnable_pos_emb=True):
        """
        class for word embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model, padding_idx)
        if learnable_pos_emb:
            self.pos_emb = LearnablePositionalEncoding(d_model, max_len)
        else:
            self.pos_emb = SinusoidalPositionalEncoding(d_model, max_len)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x).to(tok_emb.device)
        return self.drop_out(tok_emb + pos_emb)


class TokenEmbedding(nn.Embedding):
    """
    Token Embedding using torch.nn
    they will dense representation of word using weighted matrix
    """

    def __init__(self, vocab_size, d_model, padding_idx):
        """
        class for token embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=padding_idx)


class SinusoidalPositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """

    def __init__(self, d_model, max_len):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length

        """
        super(SinusoidalPositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len = x.size()
        # [batch_size = 128, seq_len = 30]

        return self.encoding[:seq_len, :]
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]


class LearnablePositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """

    def __init__(self, d_model, max_seq_len):
        """
        constructor of learnable positonal encoding class

        :param d_model: dimension of model
        :param max_seq_len: max sequence length

        """
        super(LearnablePositionalEncoding, self).__init__()
        self.max_seq_len = max_seq_len
        self.wpe = nn.Embedding(max_seq_len, d_model)

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]
        device = x.device
        batch_size, seq_len = x.size()
        assert seq_len <= self.max_seq_len, f"Cannot forward sequence of length {seq_len}, max_seq_len is {self.max_seq_len}"
        pos = torch.arange(0, seq_len, dtype=torch.long, device=device) # shape (seq_len)
        pos_emb = self.wpe(pos) # position embeddings of shape (seq_len, d_model)

        return pos_emb
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]


class Encoder(nn.Module):

    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, padding_idx, learnable_pos_emb=True):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model,
                                        max_len=max_len,
                                        vocab_size=enc_voc_size,
                                        drop_prob=drop_prob,
                                        padding_idx=padding_idx,
                                        learnable_pos_emb=learnable_pos_emb
                                        )

        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    def forward(self, x, s_mask):
        x = self.emb(x)

        for layer in self.layers:
            x = layer(x, s_mask)

        return x

class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, padding_idx, learnable_pos_emb=True):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model,
                                        drop_prob=drop_prob,
                                        max_len=max_len,
                                        vocab_size=dec_voc_size,
                                        padding_idx=padding_idx,
                                        learnable_pos_emb=learnable_pos_emb
                                        )

        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        trg = self.emb(trg)

        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)

        # pass to LM head
        output = self.linear(trg)
        return output

class Transformer(nn.Module):

    def __init__(self, src_pad_idx, trg_pad_idx, enc_voc_size, dec_voc_size, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, learnable_pos_emb=True):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               enc_voc_size=enc_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               padding_idx=src_pad_idx,
                               learnable_pos_emb=learnable_pos_emb)

        self.decoder = Decoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               dec_voc_size=dec_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               padding_idx=trg_pad_idx,
                               learnable_pos_emb=learnable_pos_emb)

    def get_device(self):
        return next(self.parameters()).device

    def forward(self, src, trg):
        device = self.get_device()
        src_mask = self.make_pad_mask(src, src, self.src_pad_idx, self.src_pad_idx).to(device)
        src_trg_mask = self.make_pad_mask(trg, src, self.trg_pad_idx, self.src_pad_idx).to(device)
        trg_mask = self.make_pad_mask(trg, trg, self.trg_pad_idx, self.trg_pad_idx).to(device) * \
                   self.make_no_peak_mask(trg, trg).to(device)

        #print(src_mask)
        #print('-'*100)
        #print(trg_mask)
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_trg_mask)
        return output

    def make_pad_mask(self, q, k, q_pad_idx, k_pad_idx):
        len_q, len_k = q.size(1), k.size(1)

        # batch_size x 1 x 1 x len_k
        k = k.ne(k_pad_idx).unsqueeze(1).unsqueeze(2)
        # batch_size x 1 x len_q x len_k
        k = k.repeat(1, 1, len_q, 1)

        # batch_size x 1 x len_q x 1
        q = q.ne(q_pad_idx).unsqueeze(1).unsqueeze(3)
        # batch_size x 1 x len_q x len_k
        q = q.repeat(1, 1, 1, len_k)

        mask = k & q
        return mask

    def make_no_peak_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)

        # len_q x len_k
        mask = torch.tril(torch.ones(len_q, len_k)).type(torch.BoolTensor)

        return mask


def make_pad_mask(x, pad_idx):
    q = k = x
    q_pad_idx = k_pad_idx = pad_idx
    len_q, len_k = q.size(1), k.size(1)

    # batch_size x 1 x 1 x len_k
    k = k.ne(k_pad_idx).unsqueeze(1).unsqueeze(2)
    # batch_size x 1 x len_q x len_k
    k = k.repeat(1, 1, len_q, 1)

    # batch_size x 1 x len_q x 1
    q = q.ne(q_pad_idx).unsqueeze(1).unsqueeze(3)
    # batch_size x 1 x len_q x len_k
    q = q.repeat(1, 1, 1, len_k)

    mask = k & q
    return mask



# x_list is a list of tensors of shape TxH where T is the seqlen and H is the feats dim
def pad_seq_v2(sequences, batch_first=True, padding_value=0.0, prepadding=True):
    lens = [i.shape[0]for i in sequences]
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=padding_value) # NxTxH
    if prepadding:
        for i in range(len(lens)):
            padded_sequences[i] = padded_sequences[i].roll(-lens[i])
    if not batch_first:
        padded_sequences = padded_sequences.transpose(0, 1) # TxNxH
    return padded_sequences



if __name__ == '__main__':
    import torch
    import random
    import numpy as np

    rand_seed = 10

    device = 'cpu'

    # model parameter setting
    batch_size = 128
    max_len = 256
    d_model = 512
    n_layers = 3
    n_heads = 16
    ffn_hidden = 2048
    drop_prob = 0.1

    # optimizer parameter setting
    init_lr = 1e-5
    factor = 0.9
    adam_eps = 5e-9
    patience = 10
    warmup = 100
    epoch = 1000
    clip = 1.0
    weight_decay = 5e-4
    inf = float('inf')

    src_pad_idx = 2
    trg_pad_idx = 3

    enc_voc_size = 37
    dec_voc_size = 15
    model = Transformer(src_pad_idx=src_pad_idx,
                        trg_pad_idx=trg_pad_idx,
                        d_model=d_model,
                        enc_voc_size=enc_voc_size,
                        dec_voc_size=dec_voc_size,
                        max_len=max_len,
                        ffn_hidden=ffn_hidden,
                        n_head=n_heads,
                        n_layers=n_layers,
                        drop_prob=drop_prob
                        ).to(device)

    random.seed(rand_seed)
    # Set the seed to 0 for reproducible results
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)

    x_list = [
        torch.tensor([[1, 1]]).transpose(0, 1), # 2
        torch.tensor([[1, 1, 1, 1, 1, 1, 1]]).transpose(0, 1),  # 7
        torch.tensor([[1, 1, 1]]).transpose(0, 1) # 3
    ]


    src_pad_idx = model.src_pad_idx
    trg_pad_idx = model.trg_pad_idx

    src = pad_seq_v2(x_list, padding_value=src_pad_idx, prepadding=False).squeeze(2)
    trg = pad_seq_v2(x_list, padding_value=trg_pad_idx, prepadding=False).squeeze(2)
    out = model(src, trg)
#######

def pad_seq(sequences, batch_first=True, padding_value=0.0, prepadding=True):
    lens = [i.shape[0] for i in sequences]
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=padding_value)
    if prepadding:
        for i in range(len(lens)):
            padded_sequences[i] = padded_sequences[i].roll(-lens[i])
    if not batch_first:
        padded_sequences = padded_sequences.transpose(0, 1)
    return padded_sequences


def get_batches(X, batch_size=16):
    num_batches = math.ceil(len(X) / batch_size)
    for i in range(num_batches):
        yield X[i * batch_size: (i + 1) * batch_size]


class TashkeelModel(pl.LightningModule):
    def __init__(self, tokenizer, max_seq_len, d_model=512, n_layers=3, n_heads=16, drop_prob=0.1, learnable_pos_emb=True):
        super(TashkeelModel, self).__init__()

        ffn_hidden = 4 * d_model
        src_pad_idx = tokenizer.letters_map['<PAD>']
        trg_pad_idx = tokenizer.tashkeel_map['<PAD>']
        enc_voc_size = len(tokenizer.letters_map)
        dec_voc_size = len(tokenizer.tashkeel_map)

        self.transformer = Transformer(
            src_pad_idx=src_pad_idx,
            trg_pad_idx=trg_pad_idx,
            d_model=d_model,
            enc_voc_size=enc_voc_size,
            dec_voc_size=dec_voc_size,
            max_len=max_seq_len,
            ffn_hidden=ffn_hidden,
            n_head=n_heads,
            n_layers=n_layers,
            drop_prob=drop_prob,
            learnable_pos_emb=learnable_pos_emb
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.tashkeel_map['<PAD>'])
        self.tokenizer = tokenizer

    def forward(self, x, y=None):
        return self.transformer(x, y)

    def training_step(self, batch, batch_idx):
        input_ids, target_ids = batch
        input_ids = input_ids[:, :-1]
        y_in = target_ids[:, :-1]
        y_out = target_ids[:, 1:]
        y_pred = self(input_ids, y_in)
        loss = self.criterion(y_pred.transpose(1, 2), y_out)

        self.log('train_loss', loss, prog_bar=True)
        sch = self.lr_schedulers()
        sch.step()
        self.log('lr', sch.get_last_lr()[0], prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, target_ids = batch
        input_ids = input_ids[:, :-1]
        y_in = target_ids[:, :-1]
        y_out = target_ids[:, 1:]

        y_pred = self(input_ids, y_in)
        loss = self.criterion(y_pred.transpose(1, 2), y_out)

        pred_texts = self.tokenizer.decode(input_ids, y_pred.argmax(2))
        true_texts = self.tokenizer.decode(input_ids, y_out)

        distance = 0
        ref_length = 0
        for pred, true in zip(pred_texts, true_texts):
            der = self.tokenizer.compute_der(true, pred)
            distance += der["distance"]
            ref_length += der["ref_length"]

        return {
            "val_loss": loss.detach(),
            "der_distance": torch.tensor(distance),
            "der_ref_length": torch.tensor(ref_length),
        }

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.validation_outputs).mean()
        self.log('val_loss', avg_loss)
        self.validation_outputs.clear()

    def test_step(self, batch, batch_idx):
        input_ids, target_ids = batch
        y_pred = self(input_ids, None)
        loss = self.criterion(y_pred.transpose(1, 2), target_ids)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4)
        gamma = 1 / 1.000001
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    @torch.no_grad()
    def do_tashkeel_batch(self, texts, batch_size=16, verbose=True):
        self.eval()
        device = next(self.parameters()).device
        text_with_tashkeel = []
        data_iter = get_batches(texts, batch_size)
        if verbose:
            data_iter = tqdm(data_iter, total=math.ceil(len(texts) / batch_size))

        for texts_mini in data_iter:
            input_ids_list = []
            for text in texts_mini:
                input_ids, _ = self.tokenizer.encode(text, test_match=False)
                input_ids_list.append(input_ids)

            batch_input_ids = pad_seq(
                input_ids_list,
                batch_first=True,
                padding_value=self.tokenizer.letters_map['<PAD>'],
                prepadding=False
            )

            target_ids = torch.LongTensor([[self.tokenizer.tashkeel_map['<BOS>']]] * len(texts_mini)).to(device)
            src = batch_input_ids.to(device)

            src_mask = self.transformer.make_pad_mask(src, src, self.transformer.src_pad_idx, self.transformer.src_pad_idx).to(device)
            enc_src = self.transformer.encoder(src, src_mask)

            for i in range(src.shape[1] - 1):
                trg = target_ids
                src_trg_mask = self.transformer.make_pad_mask(trg, src, self.transformer.trg_pad_idx, self.transformer.src_pad_idx).to(device)
                trg_mask = self.transformer.make_pad_mask(trg, trg, self.transformer.trg_pad_idx, self.transformer.trg_pad_idx).to(device) * \
                           self.transformer.make_no_peak_mask(trg, trg).to(device)

                preds = self.transformer.decoder(trg, enc_src, trg_mask, src_trg_mask)
                target_ids = torch.cat([target_ids, preds[:, -1].argmax(1).unsqueeze(1)], axis=1)

                # Force prediction for input space character to output <NT> tag
                target_ids[self.tokenizer.letters_map[' '] == src[:, :target_ids.shape[1]]] = \
                    self.tokenizer.tashkeel_map[self.tokenizer.no_tashkeel_tag]

            text_with_tashkeel_mini = self.tokenizer.decode(src, target_ids)
            text_with_tashkeel += text_with_tashkeel_mini

        return text_with_tashkeel

    @torch.no_grad()
    def do_tashkeel(self, text):
        return self.do_tashkeel_batch([text])[0]
