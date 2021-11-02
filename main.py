from nltk.sem import relextract
from nltk.corpus import ieer

docs = ieer.parsed_docs('NYT_19980315')
tree = docs[1].text
#print(tree)

pairs = relextract.tree2semi_rel(tree)
# for s, tree in pairs[18:22]:
#     print('("...%s", %s)' % (" ".join(s[-5:]),tree))

reldicts = relextract.semi_rel2reldict(pairs)
for k, v in sorted(reldicts[0].items()):
     print(k, '=>', v)

for r in reldicts:
     print('=' * 20)
     print(r['subjtext'])
     print(r['filler'])
     print(r['objtext'])
