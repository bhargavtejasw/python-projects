model_resnet18 = models.resnet18(pretrained=True)
model_resnet18.fc = nn.Linear(512, output_label)
model_resnet18.to(device)
model_resnet18
max_lr = 0.0001
epoch = 20
weight_decay = 1e-4
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_resnet18.parameters(), lr=max_lr, weight_decay=weight_decay)
sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epoch, 
                                            steps_per_epoch=len(train_loader))
history_re18 = fit(epoch, model_resnet18, train_loader, val_loader, criterion, optimizer, sched)
torch.save(model_resnet18.state_dict(),'resnet18.pth')
plot_score(history_re18, epoch)
plot_loss(history_re18, epoch)
plot_lr(history_re18)
def predict_dataset(dataset, model):
    model.eval()
    model.to(device)
    torch.cuda.empty_cache()
    predict = []
    y_true = []
    for image, label in dataset:
        #image = image.to(device); label= label.to(device)
        image = image.unsqueeze(0)
        image = image.to(device);
        
        output = model(image)
        ps = torch.exp(output)
        _, top_class = ps.topk(1, dim=1)
        
        predic = np.squeeze(top_class.cpu().numpy())
        predict.append(predic)
        y_true.append(label)
    return list(y_true), list(np.array(predict).reshape(1,-1).squeeze(0))
def report(y_true, y_predict, title='MODEL OVER TEST SET'):
    print(classification_report(y_true, y_predict))
    sns.heatmap(confusion_matrix(y_true, y_predict), annot=True)
    plt.yticks(np.arange(0.5, len(TARGET_LABEL)), labels=list(TARGET_LABEL.values()), rotation=0);
    plt.xticks(np.arange(0.5, len(TARGET_LABEL)), labels=list(TARGET_LABEL.values()), rotation=45)
    plt.title(title)
    plt.show()
    
def plot_predict(test_set, y_predict):
    """it takes longer time to plot, if you want it faster
    comment or delete tight_layout
    """
    fig = plt.figure(figsize=(20, 20))
for i in range(len(test_set)):
        image, label = test_set[i]
        ax = fig.add_subplot(4, 6, i+1, xticks=[], yticks = [])
        ax.imshow(image[0], cmap='gray')
        ax.set_title("{}({})" .format(TARGET_LABEL[y_predict[i]], TARGET_LABEL[label]), 
                      color=("green" if y_predict[i] == label else 'red'), fontsize=12)
plt.tight_layout() #want faster comment or delete this
    plt.show()
y_true, y_predict = predict_dataset(test_set, model_mobile)
report(y_true, y_predict, title='Mobilenet_v2 Over Test Set')
plot_predict(test_set, y_predict)
y_true, y_predict = predict_dataset(test_set, model_resnet18)
report(y_true, y_predict, 'Resnet18')
plot_predict(test_set, y_predict)