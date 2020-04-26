import numpy as np
import torch
from tqdm import tqdm


def calc_gt_grad(ref_model, Xs):
    X_withg = torch.autograd.Variable(Xs, requires_grad=True)
    score = ref_model(X_withg).max(1)[0].mean()
    score.backward()
    grad = X_withg.grad.data
    return grad


def epoch_train_2d(model, ref_model, optimizer, dataloader, total=None, N_b = 100):
    cos_fn = torch.nn.CosineSimilarity(dim=-1)
    model.train()

    tot_num = 0.0
    cum_loss = 0.0
    cum_cos1 = 0.0
    cum_cos2 = 0.0
    cum_cosinter = 0.0
    cum_lcos = 0.0
    cum_lreg = 0.0
    with tqdm(enumerate(dataloader), total=total) as pbar:
        for i, (Xs, _) in pbar:
            if next(model.parameters()).is_cuda:
                Xs = Xs.cuda()
            B = Xs.size(0)
            grad_gt = calc_gt_grad(ref_model, Xs)

            v1 = np.random.randint(N_b)
            v2 = np.random.randint(N_b)
            while (v1 == v2):
                v2 = np.random.randint(N_b)
            grad_b1, grad_b2 = model(Xs, [v1,v2])
            cos_g1 = cos_fn(grad_gt.view(B,-1), grad_b1.view(B,-1))
            cos_g2 = cos_fn(grad_gt.view(B,-1), grad_b2.view(B,-1))
            cos_inter = cos_fn(grad_b1.view(B,-1), grad_b2.view(B,-1))
            l = -( (cos_g1**2 + cos_g2**2 - 2*cos_g1*cos_g2*cos_inter) / (2-cos_inter**2) ).mean()

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            tot_num += B
            cum_loss = cum_loss + l.item() * B
            cum_cos1 = cum_cos1 + cos_g1.abs().sum().item()
            cum_cos2 = cum_cos2 + cos_g2.abs().sum().item()
            cum_cosinter = cum_cosinter + cos_inter.sum().item()
            pbar.set_description("Cur l: %.6f; Avg l: %.6f; Avg (c1/c2/ci): (%.6f, %.6f, %.6f)"%(l.item(), cum_loss / tot_num, cum_cos1 / tot_num, cum_cos2 / tot_num, cum_cosinter / tot_num))


def epoch_train_2d_all(model, ref_model, optimizer, dataloader, total=None, N_b = 100):
    cos_fn = torch.nn.CosineSimilarity(dim=-1)
    model.train()

    tot_num = 0.0
    cum_loss = 0.0
    cum_cos1 = 0.0
    cum_cos2 = 0.0
    cum_cosinter = 0.0
    cum_lcos = 0.0
    cum_lreg = 0.0
    with tqdm(enumerate(dataloader), total=total) as pbar:
        for i, (Xs, _) in pbar:
            if next(model.parameters()).is_cuda:
                Xs = Xs.cuda()
            B = Xs.size(0)
            grad_gt = calc_gt_grad(ref_model, Xs)

            grad_bs = model.gen_all_vs(Xs)
            v_order = np.random.permutation(N_b)
            l = 0
            for st in range(0,N_b-2,2):
                grad_b1 = grad_bs[v_order[st]]
                grad_b2 = grad_bs[v_order[st+1]]
                cos_g1 = cos_fn(grad_gt.view(B,-1), grad_b1.view(B,-1))
                cos_g2 = cos_fn(grad_gt.view(B,-1), grad_b2.view(B,-1))
                cos_inter = cos_fn(grad_b1.view(B,-1), grad_b2.view(B,-1))

                cur_l = -( (cos_g1**2 + cos_g2**2 - 2*cos_g1*cos_g2*cos_inter) / (2-cos_inter**2) ).mean()
                l = l + cur_l / (N_b//2)

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            tot_num += B
            cum_loss = cum_loss + l.item() * B
            cum_cos1 = cum_cos1 + cos_g1.abs().sum().item()
            cum_cos2 = cum_cos2 + cos_g2.abs().sum().item()
            cum_cosinter = cum_cosinter + cos_inter.sum().item()
            pbar.set_description("Cur l: %.6f; Avg l: %.6f; Avg (c1/c2/ci): (%.6f, %.6f, %.6f)"%(l.item(), cum_loss / tot_num, cum_cos1 / tot_num, cum_cos2 / tot_num, cum_cosinter / tot_num))
            

def epoch_eval_2d(model, ref_model, dataloader, total=None, N_b=100):
    cos_fn = torch.nn.CosineSimilarity(dim=-1)
    model.eval()

    tot_num = 0.0
    cum_loss = 0.0
    cum_cos1 = 0.0
    cum_cos2 = 0.0
    cum_cosinter = 0.0
    with tqdm(enumerate(dataloader), total=total) as pbar:
        for i, (Xs, _) in pbar:
            if next(model.parameters()).is_cuda:
                Xs = Xs.cuda()
            B = Xs.size(0)
            grad_gt = calc_gt_grad(ref_model, Xs)

            with torch.no_grad():
                v1 = np.random.randint(N_b)
                v2 = np.random.randint(N_b)
                while (v1 == v2):
                    v2 = np.random.randint(N_b)
                grad_b1, grad_b2 = model(Xs, [v1,v2])
                cos_g1 = cos_fn(grad_gt.view(B,-1), grad_b1.view(B,-1))
                cos_g2 = cos_fn(grad_gt.view(B,-1), grad_b2.view(B,-1))
                cos_inter = cos_fn(grad_b1.view(B,-1), grad_b2.view(B,-1))
                l = -( (cos_g1**2 + cos_g2**2 - 2*cos_g1*cos_g2*cos_inter) / (1-cos_inter**2) ).mean()

            tot_num += B
            cum_loss = cum_loss + l.item() * B
            cum_cos1 = cum_cos1 + cos_g1.abs().sum().item()
            cum_cos2 = cum_cos2 + cos_g2.abs().sum().item()
            cum_cosinter = cum_cosinter + cos_inter.sum().item()
            pbar.set_description("Avg l: %.6f; Avg (c1/c2/ci): (%.6f, %.6f, %.6f)"%(cum_loss / tot_num, cum_cos1 / tot_num, cum_cos2 / tot_num, cum_cosinter / tot_num))


def epoch_train_reg(model, ref_model, optimizer, dataloader, LMD, total=None, N_b=100):
    cos_fn = torch.nn.CosineSimilarity(dim=-1)
    model.train()

    tot_num = 0.0
    cum_loss = 0.0
    cum_lcos = 0.0
    cum_lreg = 0.0
    with tqdm(enumerate(dataloader), total=total) as pbar:
        for i, (Xs, _) in pbar:
            if next(model.parameters()).is_cuda:
                Xs = Xs.cuda()
            B = Xs.size(0)
            grad_gt = calc_gt_grad(ref_model, Xs)

            vs = model.gen_all_vs(Xs)
            vs_flat = vs.view(N_b,B,-1)
            #l_cos = (cos_fn(vs_flat, grad_gt.view(1,B,-1))**2).mean()
            l_cos = (cos_fn(vs_flat, grad_gt.view(1,B,-1))).mean()
            idt = torch.eye(N_b).cuda()
            vs_flat_normed = vs_flat / torch.sqrt((vs_flat**2).sum(2)).unsqueeze(2)
            vs_flat_normed = vs_flat_normed.transpose(0,1)
            l_reg = ((torch.bmm(vs_flat_normed, vs_flat_normed.transpose(1,2)) - idt)**2).mean()
            l = -l_cos + LMD * l_reg

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            tot_num += B
            cum_loss += l.item() * B
            cum_lcos += l_cos.item() * B
            cum_lreg += l_reg.item() * B
            #pbar.set_description("Cur l: %.6f; Avg l: %.6f; Avg (lcos/lreg): (%.6f, %.6f)"%(l.item(), cum_loss / tot_num, cum_lcos / tot_num, cum_lreg / tot_num))
            pbar.set_description("Cur l: %.6f; Avg l: %.6f; Cur (lcos/lreg): (%.6f, %.6f)"%(l.item(), cum_loss / tot_num, l_cos.item(), l_reg.item()))


def epoch_eval_reg(model, ref_model, dataloader, LMD, total=None, N_b=100):
    cos_fn = torch.nn.CosineSimilarity(dim=-1)
    model.eval()

    tot_num = 0.0
    cum_loss = 0.0
    cum_lcos = 0.0
    cum_lreg = 0.0
    with tqdm(enumerate(dataloader), total=total) as pbar:
        for i, (Xs, _) in pbar:
            if next(model.parameters()).is_cuda:
                Xs = Xs.cuda()
            B = Xs.size(0)
            grad_gt = calc_gt_grad(ref_model, Xs)

            with torch.no_grad():
                vs = model.gen_all_vs(Xs)
                vs_flat = vs.view(N_b,B,-1)
                #l_cos = (cos_fn(vs_flat, grad_gt.view(1,B,-1))**2).mean()
                l_cos = (cos_fn(vs_flat, grad_gt.view(1,B,-1))).mean()
                idt = torch.eye(N_b).cuda()
                vs_flat_normed = vs_flat / torch.sqrt((vs_flat**2).sum(2)).unsqueeze(2)
                vs_flat_normed = vs_flat_normed.transpose(0,1)
                l_reg = ((torch.bmm(vs_flat_normed, vs_flat_normed.transpose(1,2)) - idt)**2).mean()
                l = -l_cos + LMD * l_reg

            tot_num += B
            cum_loss += l.item() * B
            cum_lcos += l_cos.item() * B
            cum_lreg += l_reg.item() * B
            pbar.set_description("    Avg l: %.6f; Avg (lcos/lreg): (%.6f, %.6f)"%(cum_loss / tot_num, cum_lcos / tot_num, cum_lreg / tot_num))


def epoch_train_3d(model, ref_model, optimizer, dataloader, total=None, N_b=100):
    cos_fn = torch.nn.CosineSimilarity(dim=-1)
    model.train()

    tot_num = 0.0
    cum_loss = 0.0
    cum_costgt = 0.0
    cum_cosinter = 0.0
    with tqdm(enumerate(dataloader), total=total) as pbar:
        for i, (Xs, _) in pbar:
            if next(model.parameters()).is_cuda:
                Xs = Xs.cuda()
            B = Xs.size(0)
            grad_gt = calc_gt_grad(ref_model, Xs)

            v1 = np.random.randint(N_b)
            v2 = np.random.randint(N_b)
            while (v1 == v2):
                v2 = np.random.randint(N_b)
            v3 = np.random.randint(N_b)
            while (v1 == v3 or v2 == v3):
                v3 = np.random.randint(N_b)

            grad_b1, grad_b2, grad_b3 = model(Xs, [v1,v2,v3])
            c1x = cos_fn(grad_gt.view(B,-1), grad_b1.view(B,-1))
            c2x = cos_fn(grad_gt.view(B,-1), grad_b2.view(B,-1))
            c3x = cos_fn(grad_gt.view(B,-1), grad_b3.view(B,-1))
            c12 = cos_fn(grad_b1.view(B,-1), grad_b2.view(B,-1))
            c13 = cos_fn(grad_b1.view(B,-1), grad_b3.view(B,-1))
            c23 = cos_fn(grad_b2.view(B,-1), grad_b3.view(B,-1))
            proj = (c1x**2 + c2x**2 + c3x**2 - 2*c1x*c2x*c12 - 2*c1x*c3x*c13 - 2*c2x*c3x*c23 - (c1x*c23)**2 - (c2x*c13)**2 - (c3x*c12)**2 + 2*c1x*c2x*c13*c23 + 2*c1x*c3x*c12*c23 + 2*c2x*c3x*c12*c13)
            l = -proj.mean()

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            tot_num += B
            cum_loss = cum_loss + l.item() * B
            cum_costgt = cum_costgt + (c1x.abs() + c2x.abs() + c3x.abs()).sum().item() / 3
            cum_cosinter = cum_cosinter + (c12.abs() + c13.abs() + c23.abs()).sum().item() / 3
            pbar.set_description("Cur l: %.6f; Avg l: %.6f; Avg (ct/ci): (%.6f, %.6f)"%(l.item(), cum_loss / tot_num, cum_costgt / tot_num, cum_cosinter / tot_num))
            

def epoch_train_3d_all(model, ref_model, optimizer, dataloader, total=None, N_b=100):
    cos_fn = torch.nn.CosineSimilarity(dim=-1)
    model.train()

    tot_num = 0.0
    cum_loss = 0.0
    cum_costgt = 0.0
    cum_cosinter = 0.0
    with tqdm(enumerate(dataloader), total=total) as pbar:
        for i, (Xs, _) in pbar:
            if next(model.parameters()).is_cuda:
                Xs = Xs.cuda()
            B = Xs.size(0)
            grad_gt = calc_gt_grad(ref_model, Xs)

            grad_bs = model.gen_all_vs(Xs)
            v_order = np.random.permutation(N_b)
            l = 0
            for st in range(0,N_b-3,3):
                grad_b1 = grad_bs[v_order[st]]
                grad_b2 = grad_bs[v_order[st+1]]
                grad_b3 = grad_bs[v_order[st+2]]
                c1x = cos_fn(grad_gt.view(B,-1), grad_b1.view(B,-1))
                c2x = cos_fn(grad_gt.view(B,-1), grad_b2.view(B,-1))
                c3x = cos_fn(grad_gt.view(B,-1), grad_b3.view(B,-1))
                c12 = cos_fn(grad_b1.view(B,-1), grad_b2.view(B,-1))
                c13 = cos_fn(grad_b1.view(B,-1), grad_b3.view(B,-1))
                c23 = cos_fn(grad_b2.view(B,-1), grad_b3.view(B,-1))
                proj = (c1x**2 + c2x**2 + c3x**2 - 2*c1x*c2x*c12 - 2*c1x*c3x*c13 - 2*c2x*c3x*c23 - (c1x*c23)**2 - (c2x*c13)**2 - (c3x*c12)**2 + 2*c1x*c2x*c13*c23 + 2*c1x*c3x*c12*c23 + 2*c2x*c3x*c12*c13) / (2 - c12**2 - c13**2 - c23**2 + 2*c12*c13*c23)
                cur_l = -proj.mean()
                l = l + cur_l / (N_b//3)

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            tot_num += B
            cum_loss = cum_loss + l.item() * B
            cum_costgt = cum_costgt + (c1x.abs() + c2x.abs() + c3x.abs()).sum().item() / 3
            cum_cosinter = cum_cosinter + (c12.abs() + c13.abs() + c23.abs()).sum().item() / 3
            pbar.set_description("Cur l: %.6f; Avg l: %.6f; Avg (ct/ci): (%.6f, %.6f)"%(l.item(), cum_loss / tot_num, cum_costgt / tot_num, cum_cosinter / tot_num))
            

def epoch_eval_3d(model, ref_model, dataloader, total=None, N_b=100):
    cos_fn = torch.nn.CosineSimilarity(dim=-1)
    model.eval()

    tot_num = 0.0
    cum_loss = 0.0
    cum_costgt = 0.0
    cum_cosinter = 0.0
    cum_rho = 0.0
    with tqdm(enumerate(dataloader), total=total) as pbar:
        for i, (Xs, _) in pbar:
            if next(model.parameters()).is_cuda:
                Xs = Xs.cuda()
            B = Xs.size(0)
            grad_gt = calc_gt_grad(ref_model, Xs)

            with torch.no_grad():
                v1 = np.random.randint(N_b)
                v2 = np.random.randint(N_b)
                while (v1 == v2):
                    v2 = np.random.randint(N_b)
                v3 = np.random.randint(N_b)
                while (v1 == v3 or v2 == v3):
                    v3 = np.random.randint(N_b)

                grad_b1, grad_b2, grad_b3 = model(Xs, [v1,v2,v3])
                c1x = cos_fn(grad_gt.view(B,-1), grad_b1.view(B,-1))
                c2x = cos_fn(grad_gt.view(B,-1), grad_b2.view(B,-1))
                c3x = cos_fn(grad_gt.view(B,-1), grad_b3.view(B,-1))
                c12 = cos_fn(grad_b1.view(B,-1), grad_b2.view(B,-1))
                c13 = cos_fn(grad_b1.view(B,-1), grad_b3.view(B,-1))
                c23 = cos_fn(grad_b2.view(B,-1), grad_b3.view(B,-1))
                proj = (c1x**2 + c2x**2 + c3x**2 - 2*c1x*c2x*c12 - 2*c1x*c3x*c13 - 2*c2x*c3x*c23 - (c1x*c23)**2 - (c2x*c13)**2 - (c3x*c12)**2 + 2*c1x*c2x*c13*c23 + 2*c1x*c3x*c12*c23 + 2*c2x*c3x*c12*c13) / (1 - c12**2 - c13**2 - c23**2 + 2*c12*c13*c23)
                l = -proj.mean()

                rho = model.calc_rho(grad_gt[0].cpu().numpy(), Xs[0].cpu().numpy())

            tot_num += B
            cum_loss = cum_loss + l.item() * B
            cum_costgt = cum_costgt + (c1x.abs() + c2x.abs() + c3x.abs()).sum().item() / 3
            cum_cosinter = cum_cosinter + (c12.abs() + c13.abs() + c23.abs()).sum().item() / 3
            cum_rho = cum_rho + rho * B
            pbar.set_description("    Avg l: %.6f; Avg rho: %.6f; Avg (ct/ci): (%.6f, %.6f)"%(cum_loss / tot_num, cum_rho / tot_num, cum_costgt / tot_num, cum_cosinter / tot_num))

def epoch_train_seq(model, ref_model, optimizer, dataloader, total=None, N_b=100):
    cos_fn = torch.nn.CosineSimilarity(dim=-1)
    model.train()

    tot_num = 0.0
    cum_loss = 0.0
    with tqdm(enumerate(dataloader), total=total) as pbar:
        for i, (Xs, _) in pbar:
            if next(model.parameters()).is_cuda:
                Xs = Xs.cuda()
            B = Xs.size(0)
            grad_gt = calc_gt_grad(ref_model, Xs)
            #print (grad_gt.shape)

            grad_bs = model.gen_all_vs(Xs)
            #print (grad_bs[0].shape)
            tot_cos2 = 0.0
            for grad_b in grad_bs:
                cos_val = cos_fn(grad_gt.view(B,-1), grad_b.view(B,-1))
                tot_cos2 = tot_cos2 + cos_val**2
            #print (tot_cos2)
            l = -tot_cos2.mean()
            cos_comp_vals = []
            for grad_b in grad_bs:
                cos_val = cos_fn(grad_gt.view(B,-1), grad_b.view(B,-1))
                cos_comp_vals.append((cos_val**2).mean().item())
            cos_comp_vals = sorted(cos_comp_vals, reverse=True)

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            tot_num += B
            cum_loss = cum_loss + l.item() * B
            pbar.set_description("Cur l: %.6f; Avg l: %.6f; Cur highest comp(1/2/3):(%.6f,%.6f,%.6f)"%(l.item(), cum_loss / tot_num, cos_comp_vals[0], cos_comp_vals[1], cos_comp_vals[2]))

def epoch_eval_seq(model, ref_model, dataloader, total=None, N_b=100):
    cos_fn = torch.nn.CosineSimilarity(dim=-1)
    model.eval()

    tot_num = 0.0
    cum_loss = 0.0
    cum_cos_comp1 = 0.0
    cum_cos_comp2 = 0.0
    cum_cos_comp3 = 0.0
    with tqdm(enumerate(dataloader), total=total) as pbar:
        for i, (Xs, _) in pbar:
            if next(model.parameters()).is_cuda:
                Xs = Xs.cuda()
            B = Xs.size(0)
            grad_gt = calc_gt_grad(ref_model, Xs)
            #print (grad_gt.shape)

            with torch.no_grad():
                grad_bs = model.gen_all_vs(Xs)
                #print (grad_bs[0].shape)
                tot_cos2 = 0.0
                for grad_b in grad_bs:
                    cos_val = cos_fn(grad_gt.view(B,-1), grad_b.view(B,-1))
                    tot_cos2 = tot_cos2 + cos_val**2
                #print (tot_cos2)
                l = -tot_cos2.mean()
                cos_comp_vals = []
                for grad_b in grad_bs:
                    cos_val = cos_fn(grad_gt.view(B,-1), grad_b.view(B,-1))
                    cos_comp_vals.append((cos_val**2).mean().item())
                cos_comp_vals = sorted(cos_comp_vals, reverse=True)

            tot_num += B
            cum_loss = cum_loss + l.item() * B
            cum_cos_comp1 += cos_comp_vals[0] * B
            cum_cos_comp2 += cos_comp_vals[1] * B
            cum_cos_comp3 += cos_comp_vals[2] * B
            #pbar.set_description("Cur l: %.6f; Avg l: %.6f; Avg cos comp(1/2/3):(%.6f,%.6f,%.6f)"%(l.item(), cum_loss / tot_num, cum_cos_comp1 / tot_num, cum_cos_comp2 / tot_num, cum_cos_comp3 / tot_num))
            pbar.set_description("    Avg l: %.6f; Avg highest comp(1/2/3):(%.6f,%.6f,%.6f)"%(cum_loss / tot_num, cum_cos_comp1 / tot_num, cum_cos_comp2 / tot_num, cum_cos_comp3 / tot_num))


