import torch
import torch.nn.functional as F
def my_kl_div(logits, teacher_logits, mask):
    log_probs = F.log_softmax(logits, dim=-1)
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    
    prod_probs = teacher_probs * (log_probs - torch.log(teacher_probs))
    
    prod_probs = prod_probs[mask == 1]
    return prod_probs.mean()

def kl_div(logits, teacher_logits, mask):
    mask_loss = F.kl_div(
        F.log_softmax(logits, dim=-1),
        F.softmax(teacher_logits, dim=-1),
        reduction="none"
    )
    mask_loss = mask_loss[mask == 1]
    return mask_loss.mean()

def forward_kl(logits, teacher_logits, mask):
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    prod_probs = teacher_probs * student_logprobs
    prod_probs = prod_probs[mask == 1]
    prod_probs = -torch.sum(prod_probs, dim=0)
    return prod_probs.mean()

def reverse_kl(logits, teacher_logits, mask):
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    prod_probs = student_probs * teacher_logprobs
    prod_probs -= student_probs * student_logprobs
    prod_probs = prod_probs[mask == 1]
    prod_probs = -torch.sum(prod_probs, dim=-1).view(-1)
    return prod_probs.mean()

def symmetric_kl(logits, teacher_logits, mask, lam=0.99):
    for_kl = forward_kl(logits, teacher_logits, mask)
    rev_kl = reverse_kl(logits, teacher_logits, mask)
    distill_loss = (1-lam) * for_kl + lam * rev_kl
    return distill_loss

def js_distance(logits, teacher_logits, no_model_batch, lam=0.9):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_probs = (1-lam) * teacher_probs + lam * student_probs

    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    mixed_logprobs = torch.log(mixed_probs)

    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    prod_probs = torch.masked_fill(student_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distill_loss = lam * -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

    prod_probs = torch.masked_fill(teacher_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(teacher_probs * teacher_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distill_loss += (1-lam) * -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distill_loss
    
def tv_distance(logits, teacher_logits, no_model_batch):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    
    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)
    prod_probs = 0.5 * torch.masked_fill(torch.abs(teacher_probs - student_probs), inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distill_loss = torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distill_loss

def skewed_forward_kl(logits, teacher_logits, no_model_batch, lam=0.1):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_probs = lam * teacher_probs + (1-lam) * student_probs
    mixed_logprobs = torch.log(mixed_probs)
    
    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    prod_probs = torch.masked_fill(teacher_probs * mixed_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distill_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distill_loss

def skewed_reverse_kl(logits, teacher_logits, no_model_batch, lam=0.1):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_probs = (1-lam) * teacher_probs + lam * student_probs
    
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    mixed_logprobs = torch.log(mixed_probs)

    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    prod_probs = torch.masked_fill(student_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distill_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distill_loss

def get_distill_loss(logits, teacher_logits, loss_type, loss_mask):
    # if "sfkl" == loss_type: #dev
    #     distill_loss = skewed_forward_kl(logits, teacher_logits, no_model_batch, lam=args.skew_alpha)
    # elif "srkl" == loss_type:
    #     distill_loss = skewed_reverse_kl(logits, teacher_logits, no_model_batch, lam=args.skew_alpha)
    # elif "jsd" == loss_type:
    #     distill_loss = js_distance(logits, teacher_logits, no_model_batch)
    # elif "tvd" == loss_type:
    #     distill_loss = tv_distance(logits, teacher_logits, no_model_batch)
    # elif "fkl" == loss_type:
    if "fkl" == loss_type:
        distill_loss = forward_kl(logits, teacher_logits, loss_mask)
    elif "rkl" == loss_type:
        distill_loss = reverse_kl(logits, teacher_logits, loss_mask)
    elif "skl" == loss_type:
        distill_loss = symmetric_kl(logits, teacher_logits, loss_mask)
    else:
        raise NotImplementedError(f"Not recognized distillation type {loss_type}")
    return distill_loss