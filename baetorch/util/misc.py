def get_sample_dataloader(data_loader):
    dataiter = iter(data_loader)
    batch_data = dataiter.next()
    return batch_data[0], batch_data[1]
