import numpy as np

import matplotlib.pyplot as plt

import torch as t
from torch import nn
from torch.nn import functional as f
from torch.utils.data import DataLoader

from tqdm.notebook import tqdm, trange
from IPython.display import clear_output


class Discriminator(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4)

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 11)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = f.leaky_relu(self.conv1(x), 0.2)
        out = self.dropout(out)
        out = f.leaky_relu(self.conv2(out), 0.2)
        out = self.dropout(out)
        out = f.leaky_relu(self.conv3(out), 0.2)
        out = self.dropout(out)
        out = f.leaky_relu(self.conv4(out), 0.2)

        out = t.flatten(out, 1)

        out = f.leaky_relu(self.fc1(out), 0.2)
        out = self.dropout(out)
        out = f.leaky_relu(self.fc2(out), 0.2)
        out = self.dropout(out)
        out = f.leaky_relu(self.fc3(out), 0.2)
        out = self.fc4(out)

        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, dropout_rate=0.3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        identity = x
        out = f.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        out = f.relu(out + identity)

        return out


class Generator(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(Generator, self).__init__()

        self.fc = nn.Linear(256, 4 * 4 * 256)
        self.bn = nn.BatchNorm1d(4 * 4 * 256)
        self.dropout = nn.Dropout(dropout_rate)

        self.res_block = ResidualBlock(256, dropout_rate)

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),

            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        out = f.relu(self.bn(self.fc(x)))
        out = self.dropout(out)
        out = out.view(-1, 256, 4, 4)

        out = self.res_block(out)

        out = self.upsample(out)

        return out


class Trainer:
    def __init__(self, generator, discriminator, train, criterion,
                 g_optimizer=None, d_optimizer=None, g_scheduler=None,
                 d_scheduler=None, batch_size=32, num_epochs=50):

        # Используем по возможности CUDA
        self.device = 'cuda' if t.cuda.is_available() else 'cpu'
        self.generator = generator
        self.discriminator = discriminator

        self.epochs = num_epochs
        self.batch_size = batch_size

        # Создаем загрузчики данных для оптимизации памяти
        self.train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2)

        self.criterion = criterion

        # Если оптимизатора нет, тогда по дефолту ставим Adam
        self.g_optimizer = g_optimizer if g_optimizer is not None else t.optim.Adam(self.generator.parameters(), lr=3e-4)
        self.d_optimizer = d_optimizer if d_optimizer is not None else t.optim.Adam(self.discriminator.parameters(), lr=3e-4)

        self.g_scheduler = g_scheduler
        self.d_scheduler = d_scheduler

    def train(self, tqdm_disable=False, visualize=False):
        """
        Тренировка модели.

        :param tqdm_disable: Есть возможность выключить tqdm.
        :param visualize: Bool. Отвечает за отрисовку кривых потерь
        """
        self.generator.to(self.device)
        self.discriminator.to(self.device)

        report = {  # Будем хранить отчеты в списке
            'generator_losses': [],
            'discriminator_losses': [],
            'generator_learning_rates': [] if self.g_scheduler is None else [self.g_scheduler.get_last_lr()],
            'discriminator_learning_rates': [] if self.d_scheduler is None else [self.d_scheduler.get_last_lr()]
        }

        epochs_range = trange(self.epochs, leave=False) if not tqdm_disable else range(self.epochs)

        for _ in epochs_range:
            self.generator.train()
            self.discriminator.train()

            # Берем среднее только с последней эпохи
            gen_desc = np.nan if len(report['generator_losses']) == 0 \
                else np.mean(report['generator_losses'][len(report['generator_losses']) - len(self.train_loader):])
            disc_desc = np.nan if len(report['discriminator_losses']) == 0 \
                else np.mean(report['discriminator_losses'][len(report['discriminator_losses']) - len(self.train_loader):])

            train_iter = tqdm(self.train_loader, desc=f'Generator Loss: {gen_desc}, Discriminator Loss: {disc_desc}', leave=False) \
                if not tqdm_disable else self.train_loader

            # Основное обучение
            for real_images, digits in train_iter:
                real_images = real_images.to(self.device)
                real_labels = digits.to(self.device)

                onehot = f.one_hot(digits, num_classes=10)
                random_noise = t.randn(real_images.shape[0], 246)
                request = t.cat([onehot, random_noise], dim=-1).to(self.device)

                fake_images = self.generator(request)
                fake_labels = 10 * t.ones(fake_images.shape[0]).long().to(self.device)

                # Обучение дискриминатора
                self.g_optimizer.zero_grad()
                self.d_optimizer.zero_grad()

                output_real = self.discriminator(real_images.detach())
                d_loss_real = self.criterion(output_real, real_labels)

                output_fake = self.discriminator(fake_images.detach())
                d_loss_fake = self.criterion(output_fake, fake_labels)

                d_loss = d_loss_real + d_loss_fake + 0.1 * (output_real.norm(2) + output_fake.norm(2))
                d_loss.backward()
                self.d_optimizer.step()

                report['discriminator_losses'].append(d_loss.item())

                # Обучение генератора
                self.g_optimizer.zero_grad()

                output_fake = self.discriminator(fake_images)
                g_loss = self.criterion(output_fake, real_labels)

                g_loss.backward()
                self.g_optimizer.step()
                report['generator_losses'].append(g_loss.item())

            # Если имеется scheduler, то делаем шаг
            if self.g_scheduler is not None:
                self.g_scheduler.step()
                report['generator_learning_rates'].append(self.g_scheduler.get_last_lr())
            else:
                report['generator_learning_rates'].append(self.g_optimizer.param_groups[0]['lr'])

            if self.d_scheduler is not None:
                self.d_scheduler.step()
                report['discriminator_learning_rates'].append(self.d_scheduler.get_last_lr())
            else:
                report['discriminator_learning_rates'].append(self.d_optimizer.param_groups[0]['lr'])

        self.generator.to('cpu')
        self.discriminator.to('cpu')

        # Если tqdm не будет, то и стирать будет нечего
        if not tqdm_disable:
            # Удаляем tqdm
            clear_output(wait=True)

        if visualize:  # Нарисуем графики
            plt.figure(figsize=(16, 5))

            # Losses Graphics ##################
            plt.subplot(1, 2, 1)
            plt.title('Кривые потерь')

            plt.plot(report['generator_losses'], label='Обучение генератора')
            plt.plot(report['discriminator_losses'], label='Обучение дискриминатора')

            plt.grid(True)
            plt.legend()

            # Learning Rate Graphic ############
            plt.subplot(1, 2, 2)
            plt.title('Изменение learning rate')

            plt.plot(report['generator_learning_rates'], label='Learning Rate генератора')
            plt.plot(report['discriminator_learning_rates'], label='Learning Rate дискриминатора')

            plt.grid(True)
            plt.legend()

            time.sleep(0.2)
            plt.show()
