import matplotlib.pyplot as plt

# Open file and read data
with open('D:\project\jq\model\\breakhis_srcnn\loss.txt', 'r') as file:
    data_srcnn = file.readlines()

test_loss_srcnn = [float(line.split(':')[1].strip()) for line in data_srcnn if 'Test Loss' in line]

with open('D:\project\jq\model\\breakhis_edsr\loss.txt', 'r') as file:
    data_edsr = file.readlines()

test_loss_edsr = [float(line.split(':')[1].strip()) for line in data_edsr if 'Test Loss' in line]

with open('D:\project\jq\model\\breakhis_SRTN\loss.txt', 'r') as file:
    data_srtn = file.readlines()

test_loss_strn = [float(line.split(':')[1].strip()) for line in data_srtn if 'Test Loss' in line]


# Add Test Loss Curve
plt.plot(test_loss_srcnn, label='SRCNN Loss')
plt.plot(test_loss_edsr, label='EDSR Loss')
plt.plot(test_loss_strn, label='SRTN Loss')

#Add axis labels and titles
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve of different models on BreakHis')

# Add Legend
plt.legend()

# Display graphics
plt.show()