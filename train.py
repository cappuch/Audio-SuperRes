import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.dataset import Dataset
from utils.model import Model
import tqdm

def model_summary(model):
  print("model_summary")
  print()
  print("Layer_name"+"\t"*7+"Number of Parameters")
  print("="*100)
  model_parameters = [layer for layer in model.parameters() if layer.requires_grad]
  layer_name = [child for child in model.children()]
  j = 0
  total_params = 0
  print("\t"*10)
  for i in layer_name:
    print()
    param = 0
    try:
      bias = (i.bias is not None)
    except:
      bias = False  
    if not bias:
      param =model_parameters[j].numel()+model_parameters[j+1].numel()
      j = j+2
    else:
      param =model_parameters[j].numel()
      j = j+1
    print(str(i)+"\t"*3+str(param))
    total_params+=param
  print("="*100)
  print(f"Total Params:{total_params}")       

epochs = 15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(upscale_factor=6)

model_summary(model)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
dataset = Dataset(spec_dir='specs/', upscale_factor=6, fixed_length=256)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)


for epoch in tqdm.tqdm(range(epochs)):
    for batch in data_loader:
        lr_spec = batch['lr_spec'].to(device) # lr is synthetic noisy blurred spec lol... hr is the clean spec
        hr_spec = batch['hr_spec'].to(device) 
        optimizer.zero_grad()
        sr_spec = model(lr_spec)
        loss = criterion(sr_spec, hr_spec)
        loss.backward()
        optimizer.step()
        print(f"Loss: {loss.item()}")
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

torch.save(model.state_dict(), 'model.pth')