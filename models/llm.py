import clip
from tqdm import tqdm 
import torch


def get_model(model_name, clip_model, tokenizer=None):
    MODEL_NAME = {}
    if model_name == 'clip':
        MODEL_NAME = {
            'vision': clip_model.encode_image,
            'text': clip_model.encode_text, 
            'token': clip.tokenize
            }
    elif model_name == 'quilt':
        MODEL_NAME = {
            'vision': clip_model.encode_image,
            'text': clip_model.encode_text, 
            'token': clip.tokenize
            }
    elif model_name == 'plip':
        MODEL_NAME = {
            'vision': clip_model.get_image_features,  # Adjusted for Hugging Face CLIPModel
            'text': clip_model.get_text_features,
            'token': clip.tokenize
            }
    elif model_name == 'conch':
        MODEL_NAME = {
            'vision': clip_model.encode_image,
            'text': clip_model.encode_text, 
            'token': tokenizer
            }
    elif model_name == 'uni2':
        MODEL_NAME = {
            'vision': clip_model,
            'text': tokenizer, 
            'token': tokenizer
            }
    elif model_name == 'gigapath':
        MODEL_NAME = {
            'vision': clip_model,
            'text': tokenizer, 
            'token': tokenizer
            }
    elif model_name == 'virshow2':
        MODEL_NAME = {
            'vision': clip_model,
            'text': tokenizer, 
            'token': tokenizer
            }
    elif model_name == 'optimus1':
        MODEL_NAME = {
            'vision': clip_model,
            'text': tokenizer, 
            'token': tokenizer
            }
    
    return MODEL_NAME['vision'], MODEL_NAME['text'], MODEL_NAME['token']


def features_extraction(args, model, tokenizer, dataset, dataloader, get_position=False, img_feat_only=False):
    encode_image, encode_text, tokenizer = get_model(args.model, model, tokenizer)

    model.eval()
    if not img_feat_only: 
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            clip_weights = []
            for classname in dataset.classnames:
                # Tokenize the prompts
                classname = classname.replace('_', ' ')
                texts = [t.format(classname)  for t in dataset.template]
                if args.model == 'conch':
                    from CONCH.conch.open_clip_custom import tokenize
                    texts = tokenize(texts=texts, tokenizer=tokenizer)
                    class_embeddings = encode_text(texts).cuda()
                else:
                    texts = tokenizer(texts).cuda()
                    class_embeddings = encode_text(texts)
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                reduce = 'mean'
                if reduce=='mean':
                    class_embedding = class_embeddings.mean(dim=0)
                    class_embedding /= class_embedding.norm()
                    clip_weights.append(class_embedding)
                if reduce is None:
                    class_embeddings /= class_embeddings.norm(dim=1, keepdim=True)
                    clip_weights.append(class_embeddings)
            text_features = torch.stack(clip_weights, dim=-1).t().cuda() # Dim [num_classes, feature_dim]

    all_images, features, labels, cosine_similarities, positions = [], [], [], [], []
    if not get_position: 
        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(dataloader)):
                images, target = images.cuda(), target.cuda()
                if args.model == 'conch':
                    model = model.to('cuda')
                if args.model == 'plip':
                    image_features = encode_image(images)
                else:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=False):
                        image_features = encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                #all_images.append(images.cpu()) # Decomment to save images
                features.append(image_features.cpu())
                labels.append(target.cpu())
                if not img_feat_only:
                    cosine_similarity = image_features.to(torch.float16) @ text_features.to(torch.float16).t()
                    cosine_similarities.append(cosine_similarity.cpu())
            #all_images, features, labels = torch.cat(all_images), torch.cat(features), torch.cat(labels)
            all_images, features, labels = all_images, torch.cat(features), torch.cat(labels)
            if not img_feat_only: cosine_similarities = torch.cat(cosine_similarities)
    else: 
        x, y = [], []
        with torch.no_grad():
            for i, (images, target, position) in enumerate(tqdm(dataloader)):
                images, target = images.cuda(), target.cuda()
                if args.model == 'conch':
                    model = model.to('cuda')
                if args.model == 'plip':
                    image_features = encode_image(images)
                else:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=False):
                        image_features = encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                all_images.append(images.cpu())
                features.append(image_features.cpu())
                labels.append(target.cpu())
                if not img_feat_only:
                    cosine_similarity = image_features.to(torch.float16) @ text_features.to(torch.float16).t()
                    cosine_similarities.append(cosine_similarity.cpu())

                x.append(position[0])
                y.append(position[1])
            all_images, features, labels = torch.cat(all_images), torch.cat(features), torch.cat(labels)
            #all_images, features, labels = all_images, torch.cat(features), torch.cat(labels)
            if not img_feat_only: cosine_similarities = torch.cat(cosine_similarities)
            x, y = torch.cat(x), torch.cat(y)
        positions = [x, y]

    if img_feat_only: return all_images, features, labels, cosine_similarities, positions
    return all_images, features, text_features, labels, cosine_similarities, positions