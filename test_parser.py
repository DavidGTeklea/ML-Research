from transition_amr_parser.parse import AMRParser

# this will download “AMR3-structbart-L” into your Torch cache
parser = AMRParser.from_pretrained("AMR3-structbart-L")

def parse_amr(text):
    tok, _ = parser.tokenize(text)
    _, mac = parser.parse_sentence(tok)
    return mac.get_amr().to_penman(jamr=False, isi=True)

print(parse_amr("The toddler pushes the cart in the supermarket."))
