import time
import torch
from tqdm import tqdm
from helpers import list_of_distances, make_one_hot
import torch.nn.functional as F


def _train_or_test(model, dataloader, optimizer=None, class_specific=True, use_l1_mask=True,
                   coefs=None, log=print, ema=None, clst_k=1, sum_cls=True):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
    is_train = optimizer is not None
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    total_focal_loss = 0
    total_cluster_cost = 0
    total_separation_cost = 0
    total_avg_separation_cost = 0
    total_orth_loss = 0
    total_coherence_loss = 0
    total_loss = 0
    alpha = torch.tensor([1.0,0.5,1.5,2.0,1.5]).cuda()
    gamma = 2.0
    
    for i, data in enumerate(dataloader):
        image, label = data['bmode'], data['primus_label']
        input = image.cuda()
        target = label.cuda()

        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            output, min_distances, values = model(input)
            # cross_entropy = torch.nn.functional.cross_entropy(output, target)
            logpt = -F.cross_entropy(output, target, reduction='none', weight=alpha)
            pt = logpt.exp()
            focal_loss = -((1-pt)**gamma)*logpt
            focal_loss = focal_loss.mean()

            if class_specific:
                # Get prototypes of correct class
                prototypes_of_correct_class = torch.t(model.prototype_class_identity[:, label]).cuda()
                prototypes_of_correct_class = prototypes_of_correct_class.unsqueeze(-1)
                
                # Get slot indicators
                slots = torch.sigmoid(model.patch_select * model.temp)
                
                # =====================================================
                # CLUSTER COST (encourage matching to correct class)
                # =====================================================
                if clst_k == 1:
                    if not sum_cls:
                        # Take max over sub-patches, then max over prototypes
                        correct_class_prototype_activations = values * prototypes_of_correct_class  # [bsz, num_protos, num_slots]
                        correct_class_proto_act_max_sub_patch, _ = torch.max(correct_class_prototype_activations, dim=2)  # [bsz, num_protos]
                        correct_class_prototype_activations, _ = torch.max(correct_class_proto_act_max_sub_patch, dim=1)  # [bsz]
                    else:
                        # Sum over sub-patches, then max over prototypes
                        correct_class_prototype_activations = (values.sum(-1)) * prototypes_of_correct_class.squeeze(-1)  # [bsz, num_protos]
                        correct_class_prototype_activations, _ = torch.max(correct_class_prototype_activations, dim=1)
                    
                    cluster_cost = torch.mean(correct_class_prototype_activations)
                else:
                    # Top-k cluster cost
                    correct_class_prototype_activations = values * prototypes_of_correct_class  # [bsz, num_protos, num_slots]
                    correct_class_proto_act_max_sub_patch, _ = torch.max(correct_class_prototype_activations, dim=2)  # [bsz, num_protos]
                    top_k_correct_class_prototype_activations, _ = torch.topk(
                        correct_class_proto_act_max_sub_patch, k=clst_k, dim=1
                    )
                    cluster_cost = torch.mean(top_k_correct_class_prototype_activations)

                # =====================================================
                # SEPARATION COST (penalize matching to wrong class)
                # =====================================================
                prototypes_of_wrong_class = (1 - prototypes_of_correct_class.squeeze(-1)).unsqueeze(-1)
                
                if not sum_cls:
                    # Max over sub-patches, then max over prototypes
                    incorrect_class_prototype_activations_sub, _ = torch.max(
                        values * prototypes_of_wrong_class, dim=2
                    )  # [bsz, num_protos]
                    incorrect_class_prototype_activations, _ = torch.max(
                        incorrect_class_prototype_activations_sub, dim=1
                    )  # [bsz]
                else:
                    # Sum over sub-patches, then max over prototypes
                    incorrect_class_prototype_activations = (values.sum(-1)) * prototypes_of_wrong_class.squeeze(-1)
                    incorrect_class_prototype_activations, _ = torch.max(incorrect_class_prototype_activations, dim=1)
                
                separation_cost = torch.mean(incorrect_class_prototype_activations)

                # Average separation cost
                avg_separation_cost = torch.sum(
                    values * prototypes_of_wrong_class, dim=1
                ) / (values.shape[-1] * torch.sum(prototypes_of_wrong_class, dim=1))
                avg_separation_cost = torch.mean(avg_separation_cost)

                # =====================================================
                # ORTHOGONALITY LOSS (encourage orthogonal prototypes within class)
                # =====================================================
                prototype_normalized = F.normalize(model.prototype_vectors, p=2, dim=1)
                cur_basis_matrix = torch.squeeze(prototype_normalized)
                subspace_basis_matrix = cur_basis_matrix.reshape(
                    model.num_classes, model.num_prototypes_per_class, -1
                )  # [num_classes, num_protos_per_class, dim*num_slots]
                subspace_basis_matrix_T = torch.transpose(subspace_basis_matrix, 1, 2)  # [num_classes, dim*num_slots, num_protos_per_class]
                orth_operator = torch.matmul(subspace_basis_matrix, subspace_basis_matrix_T)  # [num_classes, num_protos_per_class, num_protos_per_class]
                I_operator = torch.eye(
                    subspace_basis_matrix.size(1), subspace_basis_matrix.size(1)
                ).cuda()  # [num_protos_per_class, num_protos_per_class]
                difference_value = orth_operator - I_operator  # [num_classes, num_protos_per_class, num_protos_per_class]
                orth_cost = torch.sum(torch.relu(torch.norm(difference_value, p=1, dim=[1, 2])))  # Scalar

                # =====================================================
                # COHERENCE LOSS (encourage similar sub-prototypes within prototype)
                # L_Coh = (1/m) * Σ_j max_{p^k_j, p^s_j ∈ p_j, p^s_j ≠ p^k_j} (1 - cos(p^k_j, p^s_j)) * I_k * I_s
                # =====================================================
                proto_norm_k = F.normalize(model.prototype_vectors, p=2, dim=1)
                # Shape: [num_prototypes, dim, num_slots]
                
                num_protos, dim, num_slots = proto_norm_k.shape
                
                # Reshape for batch computation
                proto_reshaped = proto_norm_k.permute(0, 2, 1)  # [num_protos, num_slots, dim]
                
                # Compute pairwise cosine similarity for all prototypes at once
                # [num_protos, num_slots, dim] @ [num_protos, dim, num_slots] -> [num_protos, num_slots, num_slots]
                cos_sim = torch.bmm(proto_reshaped, proto_reshaped.transpose(1, 2))
                
                # Convert to distance (dissimilarity)
                dist_matrix = 1 - cos_sim  # [num_protos, num_slots, num_slots]
                
                # Get slot indicators for all prototypes
                proto_slots = slots.squeeze(0)  # [num_protos, num_slots]
                
                # Create mask for valid pairs (both slots active)
                # [num_protos, num_slots, 1] * [num_protos, 1, num_slots] -> [num_protos, num_slots, num_slots]
                slot_mask = proto_slots.unsqueeze(2) * proto_slots.unsqueeze(1)
                
                # Mask out diagonal (self-similarity: p^k_j compared with itself)
                eye_mask = 1 - torch.eye(num_slots, device=dist_matrix.device).unsqueeze(0)
                # Shape: [1, num_slots, num_slots]
                
                # Combine masks: only consider pairs where both slots are active AND it's not self-comparison
                valid_mask = slot_mask * eye_mask  # [num_protos, num_slots, num_slots]
                
                # Apply mask to distance matrix
                masked_dist = dist_matrix * valid_mask  # [num_protos, num_slots, num_slots]
                
                # For each prototype, find the maximum dissimilarity (most dissimilar pair)
                # Reshape to [num_protos, num_slots * num_slots] for easier max computation
                max_dissimilarity_per_proto, _ = masked_dist.view(num_protos, -1).max(dim=1)
                # Shape: [num_protos]
                
                # Average over all prototypes
                coherence_loss = max_dissimilarity_per_proto.mean()

                # =====================================================
                # L1 REGULARIZATION
                # =====================================================
                if use_l1_mask:
                    l1_mask = 1 - torch.t(model.prototype_class_identity).cuda()
                    l1 = (model.last_layer.weight * l1_mask).norm(p=1)
                else:
                    l1 = model.last_layer.weight.norm(p=1)

            else:
                # Non-class-specific case
                min_distance, _ = torch.min(min_distances, dim=1)
                cluster_cost = torch.mean(min_distance)
                l1 = model.last_layer.weight.norm(p=1)
                separation_cost = torch.tensor(0.0).cuda()
                avg_separation_cost = torch.tensor(0.0).cuda()
                orth_cost = torch.tensor(0.0).cuda()
                coherence_loss = torch.tensor(0.0).cuda()

            # =====================================================
            # EVALUATION STATISTICS
            # =====================================================
            _, predicted = torch.max(output.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()
            n_batches += 1
            # total_cross_entropy += cross_entropy.item()
            total_focal_loss += focal_loss.item()
            total_cluster_cost += cluster_cost.item()
            total_separation_cost += separation_cost.item()
            total_avg_separation_cost += avg_separation_cost.item()
            total_orth_loss += orth_cost.item()
            total_coherence_loss += coherence_loss.item()
            
            # Calculate average number of active slots
            avg_number_patch = (slots >= 0.5).float().sum() / slots.shape[0]
            avg_slots = slots.mean()

        # =====================================================
        # COMPUTE GRADIENT AND OPTIMIZE
        # =====================================================
        if is_train:
            if class_specific:
                if coefs is not None:
                    loss = (
                        # coefs['crs_ent'] * cross_entropy
                        coefs['fcl'] * focal_loss
                        + coefs['clst'] * cluster_cost
                        + coefs['sep'] * separation_cost
                        + coefs['l1'] * l1
                        + coefs['orth'] * orth_cost
                        + coefs['coh'] * coherence_loss
                    )
                    total_loss += loss.item()
                else:
                    loss = focal_loss + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1 # cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1
            else:
                if coefs is not None:
                    loss = (
                        # coefs['crs_ent'] * cross_entropy
                        coefs['fcl'] * focal_loss
                        + coefs['clst'] * cluster_cost
                        + coefs['l1'] * l1
                    )
                else:
                    loss = focal_loss + 0.8 * cluster_cost + 1e-4 * l1 #cross_entropy + 0.8 * cluster_cost + 1e-4 * l1
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if ema is not None:
                ema.update(model)

        del input
        del target
        del output
        del predicted
        del min_distances

    end = time.time()

    # =====================================================
    # LOGGING
    # =====================================================
    log('\ttime: \t{0:.2f}s'.format(end - start))
    
    if is_train:
        log('\ttotal loss: \t{0:.4f}'.format(total_loss / n_batches))
    
    # log('\tcross ent: \t{0:.4f}'.format(total_cross_entropy / n_batches))
    log('\tfocal loss: \t{0}'.format(total_focal_loss / n_batches))
    log('\tcluster: \t{0:.4f}'.format(total_cluster_cost / n_batches))
    
    if class_specific:
        log('\tseparation:\t{0:.4f}'.format(total_separation_cost / n_batches))
        log('\tavg separation:\t{0:.4f}'.format(total_avg_separation_cost / n_batches))
        log('\torthogonal loss:\t{0:.4f}'.format(total_orth_loss / n_batches))
        log('\tcoherence loss:\t{0:.4f}'.format(total_coherence_loss / n_batches))
        log('\tl1: \t\t{0:.4f}'.format(model.last_layer.weight.norm(p=1).item()))
        
        # Log slot statistics
        with torch.no_grad():
            slots_display = torch.sigmoid(model.patch_select * model.temp)
            log('\tslot of prototype 0: \t{0}'.format(slots_display.squeeze()[0]))
            log('\tEstimated avg number of active slots: \t{0:.2f}'.format(avg_number_patch.item()))
            log('\tEstimated avg slots value: \t{0:.4f}'.format(avg_slots.item()))
    
    log('\taccuracy: \t{0:.2f}%'.format(n_correct / n_examples * 100))
    
    # Compute prototype diversity
    p = model.prototype_vectors.view(model.num_prototypes, -1).cpu()
    with torch.no_grad():
        p_avg_pair_dist = torch.mean(list_of_distances(p, p))
    log('\tp dist pair: \t{0:.4f}'.format(p_avg_pair_dist.item()))

    # Return loss values as dictionary
    if is_train:
        loss_values = {
            # "cross entropy Loss": coefs['crs_ent'] * total_cross_entropy / n_batches,
            "focal loss": coefs['fcl'] * total_focal_loss / n_batches,
            "clst loss": coefs['clst'] *  total_cluster_cost / n_batches,
            "sep loss": coefs['sep'] * total_separation_cost / n_batches,
            "avg separation_cost": total_avg_separation_cost / n_batches,
            "l1 loss": coefs['l1'] * model.last_layer.weight.norm(p=1).item(),
            "orth loss": coefs['orth'] *  total_orth_loss / n_batches,
            "coherence loss": coefs['coh'] * total_coherence_loss / n_batches,
            "total loss": total_loss/ n_batches,
            "acc": n_correct / n_examples * 100
        }
    else:
        loss_values = {
            # "cross entropy Loss": total_cross_entropy / n_batches,
            "focal loss": total_focal_loss / n_batches,
            "clst loss": total_cluster_cost / n_batches,
            "sep loss": total_separation_cost / n_batches,
            "avg separation_cost": total_avg_separation_cost / n_batches,
            "l1 loss": model.last_layer.weight.norm(p=1).item(),
            "orth loss": total_orth_loss / n_batches,
            "coherence loss": total_coherence_loss / n_batches,
            "acc": n_correct / n_examples * 100
        }


    return (n_correct / n_examples), loss_values


def train(model, dataloader, optimizer, class_specific=False, coefs=None, log=print, ema=None, clst_k=1, sum_cls=True):
    assert(optimizer is not None)
    log('\ttrain')
    model.train()
    return _train_or_test(
        model=model, 
        dataloader=dataloader, 
        optimizer=optimizer,
        class_specific=class_specific, 
        coefs=coefs, 
        log=log, 
        ema=ema, 
        clst_k=clst_k,
        sum_cls=sum_cls
    )


def test(model, dataloader, class_specific=False, log=print, ema=None, clst_k=1, sum_cls=True):
    log('\ttest')
    model.eval()
    return _train_or_test(
        model=model, 
        dataloader=dataloader, 
        optimizer=None,
        class_specific=class_specific, 
        log=log, 
        ema=ema, 
        clst_k=clst_k,
        sum_cls=sum_cls
    )


def last_only(model, log=print):
    """Freeze everything except last layer"""
    for p in model.features.parameters():
        p.requires_grad = False
    model.prototype_vectors.requires_grad = False
    for p in model.last_layer.parameters():
        p.requires_grad = True
    log('\tlast layer')


def warm_only(model, log=print):
    """Train features and prototypes, keep last layer trainable"""
    for p in model.features.parameters():
        p.requires_grad = True
    model.prototype_vectors.requires_grad = True
    for p in model.last_layer.parameters():
        p.requires_grad = True
    log('\twarm')


def joint(model, log=print):
    """Train all parameters"""
    for p in model.features.parameters():
        p.requires_grad = True
    model.prototype_vectors.requires_grad = True
    for p in model.last_layer.parameters():
        p.requires_grad = True
    log('\tjoint')