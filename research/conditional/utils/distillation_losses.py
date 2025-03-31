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

def symmetric_kl(logits, teacher_logits, mask, lam=0.9):
    for_kl = forward_kl(logits, teacher_logits, mask)
    rev_kl = reverse_kl(logits, teacher_logits, mask)
    distill_loss = (1-lam) * for_kl + lam * rev_kl
    return distill_loss

def js_distance(logits, teacher_logits, mask, lam=0.9):
    logits = logits[mask == 1]
    teacher_logits = teacher_logits[mask == 1]
    
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    student_probs = F.softmax(logits, dim=-1)
    mixed_probs = (1-lam) * teacher_probs + lam * student_probs

    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1)
    student_logprobs = F.log_softmax(logits, dim=-1)
    mixed_logprobs = torch.log(mixed_probs)

    prod_probs = student_probs * mixed_logprobs
    prod_probs -= student_probs * student_logprobs
    prod_probs = torch.sum(prod_probs, dim=-1)
    distill_loss = lam * -prod_probs.mean()

    prod_probs = teacher_probs * mixed_logprobs
    prod_probs -= teacher_probs * teacher_logprobs
    prod_probs = torch.sum(prod_probs, dim=-1)
    distill_loss += (1-lam) * -prod_probs.mean()
    return distill_loss


def tv_distance(logits, teacher_logits, mask):
    logits = logits[mask == 1]
    teacher_logits = teacher_logits[mask == 1]

    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    
    prod_probs = 0.5 * torch.abs(teacher_probs - student_probs)
    return prod_probs.mean()

def skewed_forward_kl(logits, teacher_logits, mask, lam=0.1):
    logits = logits[mask == 1]
    teacher_logits = teacher_logits[mask == 1]

    teacher_probs = F.softmax(teacher_logits, dim=-1)
    student_probs = F.softmax(logits, dim=-1)
    mixed_probs = lam * teacher_probs + (1-lam) * student_probs
    mixed_logprobs = torch.log(mixed_probs)

    prod_probs = teacher_probs * mixed_logprobs
    distill_loss = torch.sum(prod_probs, dim=-1).view(-1)
    return -distill_loss.mean()

def skewed_reverse_kl(logits, teacher_logits, mask, lam=0.1):
    logits = logits[mask == 1]
    teacher_logits = teacher_logits[mask == 1]

    teacher_probs = F.softmax(teacher_logits, dim=-1)
    student_probs = F.softmax(logits, dim=-1)
    mixed_probs = (1-lam) * teacher_probs + lam * student_probs
    
    student_logprobs = F.log_softmax(logits, dim=-1)
    mixed_logprobs = torch.log(mixed_probs)


    prod_probs = student_probs * mixed_logprobs
    prod_probs -= student_probs * student_logprobs
    distill_loss = torch.sum(prod_probs, dim=-1).view(-1)
    return -distill_loss.mean()


def distilbert(student_logits, teacher_logits, mask, ce_loss=1.0, temperature=2.0, alpha=5.0, beta=1.0): #dev
    student_logits = student_logits[mask == 1]
    teacher_logits = teacher_logits[mask == 1]

    # Compute softened probabilities
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    # KL(teacher || student)
    kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)

    total_loss = alpha * ce_loss + beta * kl_loss
    return total_loss

def distilbert_loss(
    student_logits,
    teacher_logits,
    mask,
    student_hidden,
    teacher_hidden,
    ce_loss,
    temperature=2.0,
    alpha=5.0,
    beta=1.0,
    gamma=2.0
):
    student_logits = student_logits[mask == 1]
    teacher_logits = teacher_logits[mask == 1]
    student_hidden = student_hidden[mask == 1]
    teacher_hidden = teacher_hidden[mask == 1]
    # Masked LM Loss (Cross-Entropy with labels)

    # Distillation loss (soft cross-entropy)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    distill_loss = F.kl_div(
        student_log_probs, 
        teacher_probs, 
        reduction='batchmean'
    ) * (temperature ** 2)

    # Cosine embedding loss (align hidden states)
    s_hidden = student_hidden.view(-1, student_hidden.size(-1))
    t_hidden = teacher_hidden.view(-1, teacher_hidden.size(-1))
    cos_targets = torch.ones(s_hidden.size(0)).to(s_hidden.device)
    cos_loss = F.cosine_embedding_loss(s_hidden, t_hidden, cos_targets)

    # Final combined loss
    total_loss = alpha * ce_loss + beta * distill_loss + gamma * cos_loss
    return total_loss

def get_distill_loss(logits, teacher_logits, loss_mask, loss_type, method_lam=None):
    if "sfkl" == loss_type: #dev
        assert method_lam
        distill_loss = skewed_forward_kl(logits, teacher_logits, loss_mask, method_lam)
    elif "srkl" == loss_type:
        assert method_lam
        distill_loss = skewed_reverse_kl(logits, teacher_logits, loss_mask, method_lam)
    elif "jsd" == loss_type:
        assert method_lam
        distill_loss = js_distance(logits, teacher_logits, loss_mask, method_lam)
    elif "tvd" == loss_type:
        assert not method_lam
        distill_loss = tv_distance(logits, teacher_logits, loss_mask)
    elif "fkl" == loss_type:
        assert not method_lam
        distill_loss = forward_kl(logits, teacher_logits, loss_mask)
    elif "rkl" == loss_type:
        assert not method_lam
        distill_loss = reverse_kl(logits, teacher_logits, loss_mask)
    elif "skl" == loss_type:
        assert method_lam
        distill_loss = symmetric_kl(logits, teacher_logits, loss_mask, method_lam)
    else:
        raise NotImplementedError(f"Not recognized distillation type {loss_type}")
    return distill_loss