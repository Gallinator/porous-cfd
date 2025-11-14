import torch
from torch import Tensor


class FoamData:
    """
    Wrapper around a Tensor which supports indexing by variable and by subdomain.

    The labels must be passed as a dict with entries according to label_name:sub_labels. If the label has no sub-labels set None.
    The dictionary order of the single labels is used for indexing. The single labels must be added before all multi labels.

    The domain must be passed as a dictionary with samples indices of shape (N) associated to each subdomain. Labels and subdomains must have different names.

    Supports batched data tensors.
    """

    def __init__(self, data: Tensor, labels: dict[str:Tensor], domain: dict[str:(list | None)]):
        """
        :param data: Tensor of shape (B,N,D) or (N,D)
        :param labels: Dictionary of labels.
        :param domain: Dictionary of sub domains.
        """
        super().__init__()
        self.data = data
        self.labels = labels
        self.domain = domain

    def make_ids_gatherable(self, flat_ids: Tensor) -> Tensor:
        """
        Allows the usage of torch.gather() on subdomain ids by repeating on last dimension.
        :param flat_ids: Ids with shape (N).
        :return: Gatherable flat_ids.
        """
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

    def __contains__(self, item):
        return item in self.domain or item in self.labels

    def squeeze(self) -> 'FoamData':
        """
        Removes the bach dimension.
        """
        squeezed_data = self.data.squeeze()
        squeezed_domain = {k: v.squeeze() for k, v in self.domain.items()}
        return FoamData(squeezed_data, self.labels, squeezed_domain)

    def to(self, *args, **kwargs) -> 'FoamData':
        """
        Calls to() on the underlying tensors.
        :return: A copy of this FoamData.
        """
        new_data = self.data.to(*args, **kwargs)
        new_domain = {d: s.to(*args, **kwargs) for d, s in self.domain.items()}
        return FoamData(new_data, self.labels, new_domain)
