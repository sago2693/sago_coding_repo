def remove_spaces_and_scape_sequences(text):
    text = text.replace('\n', ' ').strip()
    text = " ".join(text.split()) 
    return text

def extract_by_greater_lesser(html_text,substring_position):
    if html_text is None:
        return None
    else:
        #Extracts substring from html text by the position of the greater and lesser than signs
        text = str(html_text)
        extracted_text = text.split(">")[substring_position].split("<")[0]
        extracted_text = remove_spaces_and_scape_sequences(extracted_text)
        return extracted_text

def extract_by_tag(html_text, tag):
    if html_text is None:
        return None
    else:
        text = str(html_text)
        extracted_text = [ s.split('</'+ tag +'>')[0] for s in text.split('<'+tag+'>')[1:]][0]
        extracted_text = remove_spaces_and_scape_sequences(extracted_text)
        return extracted_text