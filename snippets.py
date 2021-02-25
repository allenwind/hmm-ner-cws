def bio2iobes(tags):
    # BIO标签转IOBES标签
    def split_spans(tags):
        buf = []
        for tag in tags:
            if tag == "O" or tag.startswith("B"):
                if buf:
                    yield buf
                buf = [tag]
            else:
                # tag.startswith("I")
                buf.append(tag)
        if buf:
            yield buf

    ntags = []
    for span in split_spans(tags):
        tag = span[0]
        if len(span) == 1:
            if tag == "O":
                ntags.append(tag)
            else:
                tag = "S" + tag[1:]
                ntags.append(tag)
        else:
            btag = "B" + tag[1:]
            itag = "I" + tag[1:]
            etag = "E" + tag[1:]
            span_tags = [btag] + [itag] * (len(span) - 2) + [etag]
            ntags.extend(span_tags)
    return ntags

def iobes2bio(tags):
    # IOBES标签转BIO标签
    ntags = []
    for tag in tags:
        if tag == "O":
            ntags.append(tag)
            continue
        tag, label = tag.split("-")
        if tag == "E":
            tag = "I"
        if tag == "S":
            tag = "B"
        tag = tag + "-" + label
        ntags.append(tag)
    return ntags

def find_entities(text, tags, withO=False):
    # 根据标签提取文本中的实体
    # 适合BIO和BIOES标签
    # withO是否返回O标签内容
    def segment_by_tags(text, tags):
        buf = ""
        for tag, char in zip(tags, text):
            if tag == "O":
                label = tag
            else:
                tag, label = tag.split("-")
            if tag == "B" or tag == "S":
                if buf:
                    yield buf, plabel
                buf = char
                plabel = label
            elif tag == "I" or tag == "E":
                buf += char
            elif withO and tag == "O":
                # tag == "O"
                if buf and plabel != "O":
                    yield buf, plabel
                    buf = ""
                buf += char
                plabel = label
        if buf:
            yield buf, plabel
    return list(segment_by_tags(text, tags))
