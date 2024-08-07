from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

import sys
import json
from json import encoder
import pdb 
encoder.FLOAT_REPR = lambda o: format(o, '.3f')

# set up file names and pathes
dataDir='coco-caption'
#dataType = sys.argv[1]
algName = 'fakecap'
subtypes=['results', 'evalImgs', 'eval']
annFile = sys.argv[1] # "captions_testKarpathy.json"
resFile = sys.argv[2] # "test_preds.json" 

# pdb.set_trace()
coco = COCO(annFile)
cocoRes = coco.loadRes(resFile)

# create cocoEval object by taking coco and cocoRes
cocoEval = COCOEvalCap(coco, cocoRes)

# evaluate on a subset of images by setting
# cocoEval.params['image_id'] = cocoRes.getImgIds()
# please remove this line when evaluating the full validation set
cocoEval.params['image_id'] = cocoRes.getImgIds()

# evaluate results
cocoEval.evaluate()

outfile = resFile.replace('preds', 'res')
outfile = outfile.replace('json', 'txt')

print("saving scores to: %s" % outfile)
with open(outfile, 'w') as outfile:
  for metric, score in cocoEval.eval.items():
    outfile.write( '%s: %.2f\n'%(metric, score*100) )
    