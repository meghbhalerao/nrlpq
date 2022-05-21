import torch
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional

class SingleHeadNestedLinear(nn.Linear):
	def __init__(self, nesting_list: List, num_classes=1000, **kwargs):
		super(SingleHeadNestedLinear, self).__init__(nesting_list[-1], num_classes, **kwargs)
		self.nesting_list=nesting_list
		self.num_classes=num_classes # Number of classes for classification

	def forward(self, x):
		nesting_logits = ()
		for i, num_feat in enumerate(self.nesting_list):
			if not (self.bias is None):
				logit = torch.matmul(x[:, :num_feat], (self.weight[:, :num_feat]).t()) + self.bias
			else:
				logit = torch.matmul(x[:, :num_feat], (self.weight[:, :num_feat]).t())
			nesting_logits+= (logit,)
		return nesting_logits

class MultiHeadNestedLinear(nn.Module):
	def __init__(self, nesting_list: List, num_classes=1000, **kwargs):
		super(MultiHeadNestedLinear, self).__init__()
		self.nesting_list=nesting_list
		self.num_classes=num_classes # Number of classes for classification
		for i, num_feat in enumerate(self.nesting_list):
			setattr(self, f"nesting_classifier_{i}", nn.Linear(num_feat, self.num_classes, **kwargs))		

	def forward(self, x):
		nesting_logits = ()
		for i, num_feat in enumerate(self.nesting_list):
			nesting_logits +=  (getattr(self, f"nesting_classifier_{i}")(x[:, :num_feat]),)
		return nesting_logits

		
class FixedFeatureLayer(nn.Linear):
    # This layer just takes the first "K" Features for the classification. 
    # Creating a separate layer and customized fwd pass helps to not change the base codes at all.
    def __init__(self, in_features, out_features, **kwargs):
        super(FixedFeatureLayer, self).__init__(in_features, out_features, **kwargs)

    def forward(self, x):
        if not (self.bias is None):
            out = torch.matmul(x[:, :self.in_features], self.weight.t()) + self.bias
        else:
            out = torch.matmul(x[:, :self.in_features], self.weight.t())
        return out

class NestedCELoss(nn.Module):
	def __init__(self, **kwargs):
		super(NestedCELoss, self).__init__()
		self.criterion = nn.CrossEntropyLoss(**kwargs)
	def forward(self, output, target):
		loss=0
		for o in output:
			loss+= self.criterion(o, target)
		return loss
