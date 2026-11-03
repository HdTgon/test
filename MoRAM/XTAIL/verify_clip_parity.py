"""Check the active backbone and initialized MoRAM against a common checkpoint."""
import argparse
import copy
import hashlib
import json
import torch
from torch.nn import functional as F
from lada_clip import clip
from models.moram import get_moram


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True,help='Same local OpenAI-format checkpoint used by LADA')
    parser.add_argument('--backbone',default='ViT-B-16')
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    torch.manual_seed(0)
    reference,_=clip.load(args.checkpoint,device='cpu',jit=False)
    reference=reference.float().to(args.device).eval()
    cfg=dict(checkpoint_path=args.checkpoint,backbone_type=args.backbone,pretrained_weight='openai',
             prec='fp32',device=args.device,zero_shot=True,target_encoder='all',
             target_modules=['q_proj','k_proj','v_proj','out_proj','ffn_in','ffn_out'],rank=4,topk=2)
    text=clip.tokenize(['a photo of a cat.','a photo of a dog.','a photo of an airplane.']).to(args.device)
    image=torch.randn(2,3,reference.visual.input_resolution,reference.visual.input_resolution,device=args.device)
    report={}
    digest=hashlib.sha256()
    with open(args.checkpoint,'rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    report['checkpoint_sha256']=digest.hexdigest()
    with torch.no_grad():
        ref_i=reference.encode_image(image)
        ref_t=reference.encode_text(text)
        ref_logits=reference.logit_scale.exp()*F.normalize(ref_i,dim=-1)@F.normalize(ref_t,dim=-1).T
        for zero in (True,False):
            cfg['zero_shot']=zero
            model,_,_,_=get_moram(cfg)
            im,tx=model.encode_image(image),model.encode_text(text)
            if not zero:
                im,tx=im[0],tx[0]
            logits=model.logit_scale.exp()*F.normalize(im,dim=-1)@F.normalize(tx,dim=-1).T
            torch.testing.assert_close(im,ref_i,atol=2e-4,rtol=2e-4)
            torch.testing.assert_close(tx,ref_t,atol=2e-4,rtol=2e-4)
            torch.testing.assert_close(logits,ref_logits,atol=2e-3,rtol=2e-4)
            report['zero_shot' if zero else 'initial_moram']={
                'image_max_abs':(im-ref_i).abs().max().item(),
                'text_max_abs':(tx-ref_t).abs().max().item(),
                'logits_max_abs':(logits-ref_logits).abs().max().item()}
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
