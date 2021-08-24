# 序列标注中常用标签的转换

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
    """根据标签提取文本中的实体，适合BIO和BIOES标签，
    withO是否返回O标签内容。
    """
    def segment_by_tags(text, tags):
        buf = ""
        plabel = None
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

def find_entities_chunking(tags):
    """根据标签提取文本中的实体始止位置，兼容BIO和BIOES标签。
    返回[(label, start, end),...]形式
    """
    def chunking_by_tags(tags):
        buf = None
        plabel = None
        for i, tag in enumerate(tags):
            if tag == "O":
                label = tag
            else:
                tag, label = tag.split("-")
            if tag == "B" or tag == "S":
                if buf:
                    yield (plabel, *buf)
                buf = [i, i+1]
                plabel = label
            elif tag == "I" or tag == "E":
                # for fix error tags sequence
                if buf is None:
                    buf = [i, i+1]
                    plabel = label
                # end
                buf[1] += 1
        if buf:
            yield (plabel, *buf)
    return list(chunking_by_tags(tags))

def find_words(text, tags):
    """通过SBME序列对text分词"""
    def segment_by_tags(text, tags):
        buf = ""
        for tag, char in zip(tags, text):
            # t is S or B
            if tag in ("B", "S"):
                if buf:
                    yield buf
                buf = char
            # t is M or E
            else:
                buf += char
        if buf:
            yield buf
    return list(segment_by_tags(text, tags))

def to_regions(segments):
    regions = []
    start = 0
    for word in segments:
        end = start + len(word)
        regions.append((start, end))
        start = end
    return regions

def find_words_regions(text, tags, with_word=False):
    """返回[(word, start, end), ...]或[(start, end), ...]形式"""
    segments = find_words(text, tags)
    regions = []
    start = 0
    for word in segments:
        end = start + len(word)
        if with_word:
            regions.append((word, start, end))
        else:
            regions.append((start, end))
        start = end
    return regions
