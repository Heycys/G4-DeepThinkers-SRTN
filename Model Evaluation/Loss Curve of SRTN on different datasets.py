import matplotlib.pyplot as plt

# Open file and read data
with open('D:\project\jq\model\\breakhis_SRTN\loss.txt', 'r') as file:
    data_breakhis = file.readlines()

test_loss_breakhis = [float(line.split(':')[1].strip()) for line in data_breakhis if 'Test Loss' in line]

with open('D:\project\jq\model\\tcga_SRTN\loss.txt', 'r') as file:
    data_tcga = file.readlines()

test_loss_tcga = [float(line.split(':')[1].strip()) for line in data_tcga if 'Test Loss' in line]

with open('D:\project\jq\model\large_tcga_SRTN\loss.txt', 'r') as file:
    data_tcgal = file.readlines()

test_loss_tcgal = [float(line.split(':')[1].strip()) for line in data_tcgal if 'Test Loss' in line]


# Add Test Loss Curve
plt.plot(test_loss_breakhis, label='BreakHis')
plt.plot(test_loss_tcga, label='TCGA-KIRC-mini')
plt.plot(test_loss_tcgal, label='TCGA-KIRC')

#Add axis labels and titles
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve of SRTN on different datasets')

# Add Legend
plt.legend()

# Display graphics
plt.show()
