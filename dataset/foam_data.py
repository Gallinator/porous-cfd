import torch
from torch import Tensor


class FoamData:
    def __init__(self, data: Tensor, labels: dict, domain: dict):
        super().__init__()
        self.data = data
        self.labels = labels
        self.domain = domain

    def make_ids_gatherable(self, flat_ids):
        return flat_ids.unsqueeze(dim=-1).repeat(1, 1, self.data.shape[-1])

    def __getitem__(self, item):
        if item in self.labels:
            label = self.labels[item]
            # Multi item label
            if label:
                sublabels = [self[l] for l in label]
                return torch.cat(sublabels, dim=-1)
            # Single item label
            else:
                col = list(self.labels.keys()).index(item)
                return self.data[..., col:col + 1]

        elif item in self.domain:
            domain_ids = self.domain[item]
            # Batched data
            if len(self.data.shape) > 2:
                gather_ids = self.make_ids_gatherable(domain_ids)
                subdomain_data = torch.gather(self.data, 1, gather_ids)
            else:
                subdomain_data = self.data[domain_ids]
            return FoamData(subdomain_data, self.labels,
                            {item: torch.arange(0, len(domain_ids))})
        else:
            raise KeyError(f'{item} not found in labels or subdomains.'
                           f' Available labels are {list(self.labels.keys())}. '
                           f'Available subdomains are {list(self.domain.keys())}.')

    def squeeze(self):
        squeezed_data = self.data.squeeze()
        squeezed_domain = {k: v.squeeze() for k, v in self.domain.items()}
        return FoamData(squeezed_data, self.labels, squeezed_domain)
