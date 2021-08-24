import jieba
from task_cws import tokenizer
import dataset

for text in dataset.load_sentences():
    print("jieba:", jieba.lcut(text, HMM=True))
    print("myHMM:", tokenizer.cut(text))

# ['守得云', '开见', '月', '明']
# ['守得', '云开', '见月', '明']
# ['乒乓球', '拍卖', '完', '了']
# ['乒乓球', '拍卖', '完', '了']
# ['无线电', '法国', '别', '研究']
# ['无线', '电法', '国别', '研究']
# ['广东省', '长假', '成绩单']
# ['广东', '省长', '假', '成绩', '单']
# ['欢迎', '新', '老师', '生前', '来', '就餐']
# ['欢迎', '新', '老师', '生前', '来', '就', '餐']
