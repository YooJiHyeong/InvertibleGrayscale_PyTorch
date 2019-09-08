import torch


class Runner:
    def __init__(self, encoder, decoder, loss, optim, train_loader, test_loader, config, device, tensorboard, fixed_test=True):
        self.encoder = encoder
        self.decoder = decoder
        self.loss = loss
        self.optim = optim
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.config = config
        self.device = device
        self.tensorboard = tensorboard
        self.fixed_test = fixed_test

        if self.fixed_test:
            self.fixed_input = next(self.test_loader).to(self.device["test"])

    def train(self):
        for epoch in range(self.config["epoch"]):
            for i, original_img in enumerate(self.train_loader):
                original_img = original_img.to(self.device["images"])
                # print(original_img.max(), original_img.min(), original_img.mean())
                gray_img = self.encoder(original_img)
                # print(gray_img.max(), gray_img.min(), gray_img.mean())
                restored_img = self.decoder(gray_img)
                # print(restored_img.max(), restored_img.min(), restored_img.mean())

                loss_stage = 1 if epoch < 90 else 2
                loss = self.loss(gray_img, original_img, restored_img, loss_stage)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                if i % 50 == 0:
                    print("[%03d/%03d] %d iter / Loss : %f" % (epoch, self.config["epoch"], i, loss))
                    self.test(epoch * 1000 + i)

            print("=========== Epoch : %03d Finished ============" % epoch)

    def test(self, epoch):
        original_img = next(self.test_loader).to(self.device["test"])

        if self.fixed_test:
            original_img = torch.cat([self.fixed_input, original_img], 0)

        gray_img = self.encoder(original_img)
        restored_img = self.decoder(gray_img)
        self.tensorboard.log_image(original_img, gray_img, restored_img, epoch)
