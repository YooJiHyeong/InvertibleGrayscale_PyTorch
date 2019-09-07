

class Runner:
    def __init__(self, encoder, decoder, loss, optim, train_loader, test_loader, config, device, tensorboard):
        self.encoder = encoder
        self.decoder = decoder
        self.loss = loss
        self.optim = optim
        self.train_loader = train_loader
        self.test_loader = iter(next(test_loader))

        self.config = config
        self.device = device
        self.tensorboard = tensorboard

    def train(self):
        for epoch in range(self.config["epoch"]):
            for i, original_img in enumerate(self.train_loader):
                original_img = original_img.to(self.device["output"])
                gray_img = self.encoder(original_img)
                restored_img = self.decoder(gray_img)
                loss = self.loss(gray_img, original_img, restored_img)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                if i % 100 == 0:
                    print("[%03d/%03d] %d iter / Loss : %f" % (epoch, self.config["epoch"], i, loss))

            print("=========== Epoch : %03d Finished ============" % epoch)
            self.test(epoch)

    def test(self, epoch):
        original_img = next(self.test_loader).to(self.device["output"])
        gray_img = self.encoder(original_img)
        restored_img = self.decoder(gray_img)
        self.tensorboard.log_image(original_img, gray_img, restored_img, epoch)
