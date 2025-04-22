<template>
  <div class="resources-page">
    <div class="resources-container">
      <!-- 基础阅读部分 -->
      <section id="basic-reading" class="resource-section">
        <h2>基础阅读（适合初学者）</h2>

        <div id="paper-reading" class="resource-category">
          <h3>1. 论文阅读（GAN的原始论文）</h3>

          <div class="gan-intro">
            <div class="gan-text">
              <h4>生成对抗网络(GAN)简介</h4>
              <p>生成对抗网络(Generative Adversarial Networks)是一种深度学习框架，由生成器(Generator)和判别器(Discriminator)两部分组成，通过对抗训练的方式学习数据分布。</p>
              <p>核心思想是让两个神经网络相互博弈 - 生成器试图生成逼真的假数据，判别器则试图区分真实数据和生成数据。</p>


            <div class="gan-image">
              <img src="@/assets/images/gan.png" alt="GAN结构示意图" class="gan-architecture-img">
              <p class="image-caption">GAN基本架构示意图</p>
            </div>
          </div>

          <div class="gan-text">
            <!-- 新增的GAN核心公式部分 -->

              <h4>GAN的核心数学表达</h4>
              <div class="formula-box">
                <p class="formula">min<sub>G</sub> max<sub>D</sub> V(D,G) = 𝔼<sub>x∼p<sub>data</sub>(x)</sub>[log D(x)] + 𝔼<sub>z∼p<sub>z</sub>(z)</sub>[log(1 - D(G(z)))]</p>
              </div>
              <div class="formula-explanation">
                <h6>公式解析：</h6>
                <ul>
                  <li><strong>D(x)</strong>: 判别器对真实数据x的判断概率</li>
                  <li><strong>G(z)</strong>: 生成器从噪声z生成的假数据</li>
                  <li><strong>max<sub>D</sub></strong>: 判别器试图最大化识别真实数据和生成数据的能力</li>
                  <li><strong>min<sub>G</sub></strong>: 生成器试图最小化判别器的识别能力(即让生成数据更逼真)</li>
                  <li><strong>p<sub>data</sub></strong>: 真实数据分布</li>
                  <li><strong>p<sub>z</sub></strong>: 噪声分布(通常为标准正态分布)</li>
                </ul>
                <p>这个minimax博弈的目标是找到一个纳什均衡点，此时生成器产生的数据分布与真实数据分布无法被判别器区分(p<sub>g</sub> = p<sub>data</sub>)。</p>
              </div>
            </div>
          </div>

          <ul class="resource-list">
            <li class="resource-item">
              <a href="https://arxiv.org/abs/1406.2661" target="_blank" class="resource-link">
                <span class="resource-icon">📄</span>
                《Generative Adversarial Nets》
              </a>
              <div class="paper-meta">
                <span class="authors">Ian Goodfellow, et al.</span>
                <span class="year">2014</span>
              </div>
              <p class="resource-description">GAN的开山之作，提出了生成器与判别器对抗训练的基本框架</p>
              <div class="resource-highlight">
                <h5>重点阅读内容：</h5>
                <ul>
                  <li>GAN的基本架构和训练目标</li>
                  <li>Minimax博弈的数学表述</li>
                  <li>价值函数的设计</li>
                  <li>理论证明：当生成器完美复制数据分布时达到全局最优</li>
                  <li>训练过程中的实际考虑</li>
                </ul>
              </div>
            </li>
          </ul>
        </div>

        <div id="tutorials" class="resource-category">
          <h3>2. 入门讲解（适合初学者）</h3>
          <ul class="resource-list">
            <li class="resource-item">
              <a href="https://www.bilibili.com/video/BV1JC4y1T7vZ" target="_blank" class="resource-link">
                <span class="resource-icon">🎬</span>
                吴恩达机器学习视频
              </a>
              <p class="resource-description">讲解GAN如何训练、生成假数据</p>
            </li>
            <li class="resource-item">
              <a href="https://www.bilibili.com/video/BV1q3411w7oA" target="_blank" class="resource-link">
                <span class="resource-icon">🎬</span>
                GAN详解视频教程
              </a>
              <p class="resource-description">数学推导过程详解</p>
            </li>
          </ul>
        </div>

        <div id="pytorch" class="resource-category">
          <h3>3. PyTorch 实践教程</h3>
          <h3>代码演示</h3>

          <div class="code-demo">
            <div class="code-tabs">
              <button
                  v-for="tab in tabs"
                  :key="tab.id"
                  :class="{ active: activeTab === tab.id }"
                  @click="activeTab = tab.id"
              >
                {{ tab.label }}
              </button>
            </div>

            <div class="code-content">
              <!-- 生成器代码 -->
              <div v-if="activeTab === 'generator'" class="code-block">
                <pre><code class="language-python"># 生成器网络定义
class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img</code></pre>
              </div>

              <!-- 判别器代码 -->
              <div v-if="activeTab === 'discriminator'" class="code-block">
                <pre><code class="language-python"># 判别器网络定义
class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity</code></pre>
              </div>

              <!-- 训练代码 -->
              <div v-if="activeTab === 'training'" class="code-block">
                <pre><code class="language-python"># GAN训练过程
def train_gan(generator, discriminator, dataloader, epochs, latent_dim):
    # 定义损失函数和优化器
    adversarial_loss = nn.BCELoss()
    optimizer_G = torch.optim.Adam(generator.parameters())
    optimizer_D = torch.optim.Adam(discriminator.parameters())

    for epoch in range(epochs):
        for i, (imgs, _) in enumerate(dataloader):
            # 真实和假标签
            real = torch.ones(imgs.size(0), 1)
            fake = torch.zeros(imgs.size(0), 1)

            # 训练判别器
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(imgs), real)
            z = torch.randn(imgs.size(0), latent_dim)
            fake_loss = adversarial_loss(discriminator(generator(z).detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # 训练生成器
            optimizer_G.zero_grad()
            z = torch.randn(imgs.size(0), latent_dim)
            g_loss = adversarial_loss(discriminator(generator(z))), real)
            g_loss.backward()
            optimizer_G.step()</code></pre>
              </div>
            </div>
          </div>

          <ul class="resource-list">
            <li class="resource-item">
              <a href="https://zhuanlan.zhihu.com/p/40393929" target="_blank" class="resource-link">
                <span class="resource-icon">💻</span>
                Gan的PyTorch实现实战教程
              </a>
            </li>
            <li class="resource-item">
              <a href="https://github.com/goodfeli/adversarial" target="_blank" class="resource-link">
                <span class="resource-icon">🔗</span>
                GitHub代码库
              </a>
            </li>
          </ul>
        </div>
      </section>

      <!-- 进阶拓展部分 -->
      <section id="advanced" class="resource-section">
        <h2>进阶拓展（适合深入学习者）</h2>

        <!-- 1. GAN改进版本 -->
        <div id="gan-improvements" class="resource-category">
          <h3>1. GANs的改进版本</h3>
          <div class="gan-evolution">
            <div class="gan-text">
              <h4>模型演进路线</h4>
              <p>从基础GAN到各个版本的改进主要围绕网络结构、损失函数和训练稳定性展开：</p>
              <ul>
                <li><strong>DCGAN</strong>：引入卷积层和批量归一化提升特征提取能力</li>
                <li><strong>WGAN</strong>：使用Wasserstein距离解决模式崩溃问题</li>
                <li><strong>BigGAN</strong>：大规模训练生成高质量复杂图像</li>
              </ul>
            </div>
            <br/>

              <img src="@/assets/images/Classification.png" alt="GAN架构演进" class="gan-architecture-img">
              <p class="image-caption">GAN模型的衍生版本</p>

          </div>

          <ul class="resource-list">
            <!-- DCGAN条目 -->
            <li class="resource-item">
              <a href="https://arxiv.org/abs/1511.06434" target="_blank" class="resource-link">
                <span class="resource-icon">📄</span>
                DCGAN（深度卷积GAN）
              </a>
              <div class="paper-meta">
                <span class="authors">Radford, et al.</span>
                <span class="year">2015</span>
              </div>
              <p class="resource-description">首次将CNN成功应用于GAN框架</p>
              <br/>

                <img src="@/assets/images/dcgan.png" alt="GAN架构演进" class="gan-architecture-img">
                <p class="image-caption">Dcgan的网络架构</p>

              <div class="resource-highlight">
                <h5>核心创新与优点：</h5>
                <ul>
                  <li>使用卷积层替代全连接层</li>
                  <li>引入批量归一化稳定训练</li>
                  <li>生成图像质量显著提升</li>
                  <li>成为后续模型基础架构</li>
                </ul>
              </div>
            </li>

            <!-- WGAN条目 -->
            <li class="resource-item">
              <a href="https://arxiv.org/abs/1701.07875" target="_blank" class="resource-link">
                <span class="resource-icon">📄</span>
                WGAN（Wasserstein GAN）
              </a>
              <div class="paper-meta">
                <span class="authors">Arjovsky, et al.</span>
                <span class="year">2017</span>
              </div>
              <p class="resource-description">通过Wasserstein距离解决模式崩溃问题</p>
              <div class="resource-highlight">
                <h5>核心创新与优点：</h5>
                <ul>
                  <li>使用Wasserstein距离替代JS散度</li>
                  <li>引入权重裁剪保证Lipschitz连续性</li>
                  <li>训练过程更加稳定可靠</li>
                  <li>损失函数与生成质量相关性更好</li>
                </ul>
              </div>
            </li>

            <!-- BigGAN条目 -->
            <li class="resource-item">
              <a href="https://arxiv.org/abs/1809.11096" target="_blank" class="resource-link">
                <span class="resource-icon">📄</span>
                BigGAN（大规模GAN）
              </a>
              <div class="paper-meta">
                <span class="authors">Brock, et al.</span>
                <span class="year">2018</span>
              </div>
              <p class="resource-description">大规模训练生成高质量复杂图像</p>
              <div class="resource-highlight">
                <h5>核心创新与优点：</h5>
                <ul>
                  <li>大规模批训练(大批量、大模型)</li>
                  <li>使用正交正则化提升稳定性</li>
                  <li>改进的截断技巧控制生成质量</li>
                  <li>在ImageNet上生成高质量复杂图像</li>
                </ul>
              </div>
            </li>
          </ul>
        </div>

        <!-- 2. 图像修复应用 -->
        <!-- 2. 图像修复应用 -->
        <div id="image-inpainting" class="resource-category">
          <h3>2. GAN在图像修复的应用</h3>

          <!-- 新增 Real-ESRGAN 原理部分 -->
          <div class="application-demo">
            <div class="app-text">
              <h4>Real-ESRGAN 盲图像复原原理</h4>
              <p>通过复合退化建模和对抗学习实现未知退化类型的修复：</p>
              <ul>
                <li><strong>高阶退化模拟</strong>：模糊核+噪声+JPEG压缩链式合成</li>
                <li><strong>RRDBNet生成器</strong>：残差密集块增强特征复用</li>
                <li><strong>U-Net判别器</strong>：多尺度纹理真实性判别</li>
              </ul>
            </div>
<!--            <div class="app-visual">-->
<!--              &lt;!&ndash; 可添加原理示意图 &ndash;&gt;-->
<!--              <img src="" alt="Real-ESRGAN架构图">-->
<!--            </div>-->
          </div>

          <!-- 原始部分卷积原理保持不变 -->
          <div class="application-demo">
            <div class="app-text">
              <h4>部分卷积修复原理</h4>
              <p>使用部分卷积层处理不规则缺失区域，结合GAN生成合理内容：</p>
              <ul>
                <li>动态更新掩码以跟踪已修复区域</li>
                <li>使用感知损失保持内容一致性</li>
                <li>对抗训练提升细节真实感</li>
              </ul>
            </div>
          </div>

          <div class="application-demo">
            <div class="app-text">
              <h4>扩展：U-Net在Gan的图像修复中广泛运用</h4>

              <!-- U-Net核心优势 -->
              <div class="tech-section">
                <h5>一、U-Net 的核心优势</h5>
                <ul class="feature-list">
                  <li>
                    <strong>编码器-解码器对称结构</strong>
                    <p>编码器通过卷积层逐步提取高层语义特征，解码器通过反卷积恢复细节，对称设计适合定位修复</p>
                  </li>
                  <li>
                    <strong>跳跃连接机制</strong>
                    <p>直接将编码器的低级特征（边缘、纹理）传递给解码器，避免信息丢失，确保局部细节精确生成</p>
                  </li>
                </ul>
              </div>

              <!-- U-Net与GAN协同优势 -->
              <div class="tech-section">
                <h5>二、U-Net 与 GAN 的协同优势</h5>
                <ul class="feature-list">
                  <li>
                    <strong>上下文感知能力</strong>
                    <p>全局特征理解场景语义，局部特征保证修复区域连贯性，对抗训练增强生成合理性</p>
                  </li>
                  <li>
                    <strong>复杂区域修复</strong>
                    <p>多尺度特征融合同时处理大范围缺失和精细纹理，突破传统CNN的模糊限制</p>
                  </li>
                  <li>
                    <strong>高频细节生成</strong>
                    <p>跳跃连接保留原始细节，GAN对抗损失生成毛孔级纹理，超越MSE的模糊结果</p>
                  </li>
                </ul>
              </div>
            </div>
          </div>
          <h3>图像修复相关资料</h3>
          <!-- 更新资源链接列表 -->
          <ul class="resource-list">
            <!-- 新增 Real-ESRGAN 论文 -->
            <li class="resource-item">
              <a href="https://arxiv.org/abs/2107.10833" target="_blank" class="resource-link">
                <span class="resource-icon">📄</span>
                Real-ESRGAN (ICCV 2021)
              </a>
              <p class="resource-description">通用盲图像超分辨率重建模型</p>
            </li>

            <!-- U-Net 原始论文 -->
            <li class="resource-item">
              <a href="https://arxiv.org/abs/1505.04597" target="_blank" class="resource-link">
                <span class="resource-icon">📄</span>
                U-Net (MICCAI 2015)
              </a>
              <p class="resource-description">医学图像分割的开创性架构设计</p>
            </li>

            <!-- U-Net在GAN中的应用论文 -->
            <li class="resource-item">
              <a href="https://arxiv.org/abs/1804.07723" target="_blank" class="resource-link">
                <span class="resource-icon">📄</span>
                Context Encoders (CVPR 2016)
              </a>
              <p class="resource-description">首个使用GAN处理不规则缺失区域的算法，将U-Net与GAN结合的图像修复框架</p>
            </li>


            <!-- 新增其他GAN修复论文 -->
            <li class="resource-item">
              <a href="https://arxiv.org/abs/2109.07161" target="_blank" class="resource-link">
                <span class="resource-icon">📄</span>
                GFPGAN (CVPR 2022)
              </a>
              <p class="resource-description">人脸修复与超分联合模型</p>
            </li>

            <li class="resource-item">
              <a href="https://arxiv.org/abs/2202.08013" target="_blank" class="resource-link">
                <span class="resource-icon">📄</span>
                LaMa (SIGGRAPH 2022)
              </a>
              <p class="resource-description">大掩码修复的快速GAN架构</p>
            </li>

            <!-- 原始综述论文 -->
            <li class="resource-item">
              <a href="https://arxiv.org/abs/2001.05566" target="_blank" class="resource-link">
                <span class="resource-icon">📄</span>
                图像修复综述
              </a>
              <p class="resource-description">涵盖GAN在修复领域的最新进展</p>
            </li>
          </ul>
        </div>

        <!-- 3. 其他领域应用 -->
        <div id="other-applications" class="resource-category">
          <h3>3. GAN在其他领域的应用</h3>
          <div class="application-cards">
            <div class="app-card">
              <h4>图像风格迁移</h4>
              <div class="app-card-content">
                <div class="cyclegan-intro">
                  <div class="cyclegan-text">

                    <p><b>CycleGAN核心特点</b></p>
                    <ul>
                      <li><strong>无需配对数据</strong>：学习两个域之间的映射关系而不需要精确对齐的训练样本</li>
                      <li><strong>循环一致性</strong>：确保转换是可逆的(A→B→A' ≈ A)</li>
                      <li><strong>双重结构</strong>：两个生成器(G_A→B, G_B→A)和两个判别器(D_A, D_B)</li>
                    </ul>

                    <p><b>CycleGan结构</b></p>
                    <p>CycleGAN 是一种生成对抗网络（GAN），用于图像风格转换。它的主要特点是不需要成对的数据集来实现图像转换
                      CycleGAN 由两个生成器和两个判别器组成，分别负责将图像从一种风格转换到另一种风格，并确保转换后的图像尽可能接近原始图像 。</p>

                    <p><b>生成器与判别器结构</b></p>
                    <p>生成器由编码器、残差块和解码器组成。编码器通过卷积网络将输入图像编码为特征表示
                      ，残差块用于保持图像的特征，解码器则通过反卷积将特征表示解码为输出图像 。判别器采用 PatchGAN 结构，
                      通过五层卷积网络将输入图像判别为真实或生成 。</p>

                    <p><b>损失函数</b></p>
                    <p>CycleGAN 的损失函数包括对抗损失、循环一致性损失和身份损失
                      。对抗损失用于训练生成器和判别器，使
                      生成的图像尽可能真实。循环一致性损失确保图像转换是可逆的，提高生成图像的质量。身份损失则用于保持生成图像与原始图像的相似性。</p>


                  </div>
                  <img src="@/assets/images/cyclegan模型.png" alt="CycleGAN架构" class="cyclegan-arch-img">
                    <p class="image-caption">CycleGan基本架构示意图</p>
                  <p>可参考该链接，也是上图来源<a href="https://blog.51cto.com/u_15461147/4835726 " target="_blank" class="resource-link">
                    <span class="resource-icon">📄</span>
                    【万物皆可 GAN】CycleGAN 原理详解
                  </a></p>
                </div>

                <!-- CycleGAN代码演示 -->
                <div class="code-demo">
                  <div class="code-tabs">
                    <button
                        v-for="tab in cycleganTabs"
                        :key="tab.id"
                        :class="{ active: activeCycleGANTab === tab.id }"
                        @click="activeCycleGANTab = tab.id"
                    >
                      {{ tab.label }}
                    </button>
                  </div>

                  <div class="code-content">
                    <!-- 生成器代码 -->
                    <div v-if="activeCycleGANTab === 'generator'" class="code-block">
              <pre><code class="language-python">// CycleGAN生成器结构
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        model = [
            nn.Conv2d(3, 64, 7, 1, 3, padding_mode='reflect'),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(),
        ]
        // 9个残差块
        for _ in range(9):
            model += [ResidualBlock(256)]
        // 上采样部分
        model += [
            nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 3, 7, 1, 3, padding_mode='reflect'),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)</code></pre>
                    </div>

                    <!-- 判别器代码 -->
                    <div v-if="activeCycleGANTab === 'discriminator'" class="code-block">
              <pre><code class="language-python">// CycleGAN判别器(PatchGAN)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        model = [
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 1, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 4, 1, 1)
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)</code></pre>
                    </div>

                    <!-- 损失函数 -->
                    <div v-if="activeCycleGANTab === 'loss'" class="code-block">
              <pre><code class="language-python">// CycleGAN损失函数
def compute_loss(real_A, real_B, fake_B, fake_A,
                rec_A, rec_B, D_A, D_B, G_A2B, G_B2A):
    # 对抗损失
    adv_loss_A = F.mse_loss(D_A(fake_B), torch.ones_like(D_A(fake_B)))
    adv_loss_B = F.mse_loss(D_B(fake_A), torch.ones_like(D_B(fake_A)))

    # 循环一致性损失
    cycle_loss_A = F.l1_loss(rec_A, real_A)
    cycle_loss_B = F.l1_loss(rec_B, real_B)

    # 身份损失(可选)
    identity_loss_A = F.l1_loss(G_B2A(real_A), real_A)
    identity_loss_B = F.l1_loss(G_A2B(real_B), real_B)

    # 总损失
    total_loss = adv_loss_A + adv_loss_B + \
                10.0 * (cycle_loss_A + cycle_loss_B) + \
                5.0 * (identity_loss_A + identity_loss_B)
    return total_loss</code></pre>
                    </div>

                    <!-- 训练过程 -->
                    <div v-if="activeCycleGANTab === 'training'" class="code-block">
              <pre><code class="language-python">// CycleGAN训练循环
def train_cyclegan():
    for epoch in range(epochs):
        for batch_idx, (real_A, real_B) in enumerate(dataloader):
            # 生成假图像
            fake_B = G_A2B(real_A)
            fake_A = G_B2A(real_B)

            # 重建图像
            rec_A = G_B2A(fake_B)
            rec_B = G_A2B(fake_A)

            # 判别器损失
            D_A_loss = (F.mse_loss(D_A(real_A), torch.ones_like(D_A(real_A))) +
                       F.mse_loss(D_A(fake_B.detach()), torch.zeros_like(D_A(fake_B))))

            D_B_loss = (F.mse_loss(D_B(real_B), torch.ones_like(D_B(real_B))) +
                       F.mse_loss(D_B(fake_A.detach()), torch.zeros_like(D_B(fake_A))))

            # 生成器损失
            G_loss = compute_loss(real_A, real_B, fake_B, fake_A,
                                 rec_A, rec_B, D_A, D_B, G_A2B, G_B2A)

            # 反向传播和优化
            optimizer_D.zero_grad()
            (D_A_loss + D_B_loss).backward()
            optimizer_D.step()

            optimizer_G.zero_grad()
            G_loss.backward()
            optimizer_G.step()</code></pre>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <ul class="resource-list">
              <li class="resource-item">
                <a href="https://arxiv.org/abs/1703.10593" target="_blank" class="resource-link">
                  <span class="resource-icon">📄</span>
                  CycleGAN论文
                </a>
                <p class="resource-description">非配对图像到图像转换的开创性工作</p>
              </li>
              <li class="resource-item">
                <a href="https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix" target="_blank" class="resource-link">
                  <span class="resource-icon">🔗</span>
                  PyTorch官方实现
                </a>
              </li>
            </ul>

            <div class="app-card">
              <h4>医学图像领域的GAN</h4>
              <div class="app-card-content">


                <strong> <p>1. 医学图像中的应用</p></strong>
                <ul>
                  <li><strong>去噪</strong>：GAN通过学习噪声图像与去噪图像的映射，提升图像纹理保留效果。</li>
                  <li><strong>重建</strong>：用于MRI等图像缺失数据重建，恢复因运动导致的图像信息丢失。</li>
                  <li><strong>分割</strong>：广泛应用于大脑、胸部、眼底等部位的自动分割，提升精准性。</li>
                  <li><strong>检测</strong>：在异常检测任务中，通过判别器提取潜在异常区域。</li>
                  <li><strong>分类</strong>：如SCGAN用于分类心脏图像，提升准确性，降低计算成本。</li>
                  <li><strong>合成</strong>：基于无条件或条件GAN合成医学图像，支持多模态图像转换。</li>
                </ul>

                <p><strong>2. 优势与缺点</strong></p>
                <p><strong>优势：</strong></p>
                <ul>
                  <li>能应对标注数据稀缺问题。</li>
                  <li>能处理非成对的数据，解决domain shift。</li>
                </ul>
                <p><strong>缺点：</strong></p>
                <ul>
                  <li>生成图像可信度不足，临床意义不明。</li>
                  <li>训练不稳定，易出现模式崩溃。</li>
                  <li>评价标准模糊，缺乏与临床评估一致的指标。</li>
                </ul>

                <p><strong>3. 未来展望</strong></p>
                <p>未来研究需解决GAN训练不稳定性、提高可信度并制定标准评估指标。加强与临床医生协作，是推动GAN在医学影像中应用落地的关键。</p>

                <p><b>📚 参考资料</b></p>
                <p>
                  <a href="https://zhuanlan.zhihu.com/p/59521772" target="_blank" class="resource-link">
                    <span class="resource-icon">📄</span> 知乎文章：GANs 在医学图像分析中的应用综述
                  </a>
                </p>
                <p>
                  <a href="https://arxiv.org/abs/1809.06222" target="_blank" class="resource-link">
                    <span class="resource-icon">📄</span> ArXiv 论文：GANs for Medical Image Analysis (2018)
                  </a>
                </p>
              </div>
            </div>

          </div>

        </div>

      </section>

    </div>

    <!-- 精美右侧目录 -->
    <div class="toc-container" ref="tocContainer">
      <div class="toc-header">
        <div class="toc-title">
          <svg class="toc-icon" viewBox="0 0 24 24">
            <path d="M3,6H21V8H3V6M3,11H21V13H3V11M3,16H21V18H3V16Z" />
          </svg>
          <span>文章目录</span>
        </div>
        <div class="toc-progress-bar">
          <div class="toc-progress" :style="{ height: scrollProgress + '%' }"></div>
        </div>
      </div>
      <ul class="toc-list">
        <li
            v-for="(item, index) in tocItems"
            :key="index"
            :class="{
            'toc-item': true,
            'has-children': item.children,
            'expanded': expandedItems[index],
            'current': currentSectionId === item.id,
            'child-current': isChildCurrent(item)
          }"
        >
          <div class="toc-item-main" @click="handleTocClick(index)">
            <span class="toc-indicator"></span>
            <span class="toc-text">{{ item.text }}</span>
            <span v-if="item.children" class="toc-arrow">
              <svg width="12" height="12" viewBox="0 0 24 24">
                <path d="M7 10l5 5 5-5z" fill="currentColor"/>
              </svg>
            </span>
          </div>

          <ul v-if="item.children" class="toc-sublist" :style="{ maxHeight: expandedItems[index] ? `${item.children.length * 36}px` : '0' }">
            <li
                v-for="(child, childIndex) in item.children"
                :key="childIndex"
                class="toc-subitem"
                :class="{ 'current': currentSectionId === child.id }"
                @click.stop="scrollToSection(child.id)"
            >
              <span class="toc-indicator"></span>
              <span class="toc-text">{{ child.text }}</span>
            </li>
          </ul>
        </li>
      </ul>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue';

const activeTab = ref('generator');
const tabs = [
  { id: 'generator', label: '生成器' },
  { id: 'discriminator', label: '判别器' },
  { id: 'training', label: '训练过程' }
];

const cycleganTabs=[
  { id: 'generator', label: '生成器' },
  { id: 'discriminator', label: '判别器' },
  { id: 'loss', label: '损失函数' },
  { id: 'training', label: '训练过程' }
];
const activeCycleGANTab = ref('generator');


// 目录结构
const tocItems = ref([
  {
    text: '基础阅读',
    id: 'basic-reading',
    children: [
      { text: '论文阅读', id: 'paper-reading' },
      { text: '入门讲解', id: 'tutorials' },
      { text: 'PyTorch实践', id: 'pytorch' }
    ]
  },
  {
    text: '进阶拓展',
    id: 'advanced',
    children: [
      { text: 'GAN改进版本', id: 'gan-improvements' },
      { text: '图像修复应用', id: 'image-inpainting' },
      { text: '其他领域应用', id: 'other-applications' }
    ]
  }
]);

const expandedItems = ref({});
const currentSectionId = ref('');
const scrollProgress = ref(0);
const tocContainer = ref(null);

// 初始化目录
onMounted(() => {
  // 默认展开所有有子项的目录
  tocItems.value.forEach((item, index) => {
    expandedItems.value[index] = !!item.children;
  });

  // 添加滚动监听
  window.addEventListener('scroll', handleScroll);
  handleScroll(); // 初始计算一次
});

onUnmounted(() => {
  window.removeEventListener('scroll', handleScroll);
});

// 处理目录点击
const handleTocClick = (index) => {
  const item = tocItems.value[index];
  if (item.children) {
    // 有子项则切换展开状态
    expandedItems.value[index] = !expandedItems.value[index];
  } else {
    // 无子项则直接跳转
    scrollToSection(item.id);
  }
};

// 滚动到指定章节
const scrollToSection = (id) => {
  const element = document.getElementById(id);
  if (element) {
    const headerOffset = 80;
    const elementPosition = element.getBoundingClientRect().top;
    const offsetPosition = elementPosition + window.pageYOffset - headerOffset;

    window.scrollTo({
      top: offsetPosition,
      behavior: 'smooth'
    });
  }
};

// 检查是否是当前子项
const isChildCurrent = (item) => {
  if (!item.children) return false;
  return item.children.some(child => child.id === currentSectionId.value);
};

// 滚动处理函数
const handleScroll = () => {
  // 计算滚动进度
  const scrollHeight = document.documentElement.scrollHeight - window.innerHeight;
  const scrolled = window.scrollY;
  scrollProgress.value = (scrolled / scrollHeight) * 100;

  // 查找当前可见的章节
  const sections = document.querySelectorAll('.resource-section, .resource-category');
  let currentSection = '';

  sections.forEach(section => {
    const sectionTop = section.offsetTop;
    const sectionHeight = section.offsetHeight;
    const scrollPosition = window.scrollY + 100; // 加100是为了提前触发

    if (scrollPosition >= sectionTop && scrollPosition < sectionTop + sectionHeight) {
      currentSection = section.id;
    }
  });

  if (currentSection) {
    currentSectionId.value = currentSection;

    // 自动展开包含当前子项的父项
    tocItems.value.forEach((item, index) => {
      if (item.children && item.children.some(child => child.id === currentSection)) {
        expandedItems.value[index] = true;
      }
    });
  }
};
</script>




<style scoped>

/* 公式样式 */
.gan-formula {
  margin: 20px 0;
  padding: 15px;
  background-color: #f8f9fa;
  border-radius: 5px;
  border-left: 4px solid #4285f4;
}

.formula-box {
  padding: 10px;
  background-color: white;
  border-radius: 3px;
  margin: 10px 0;
  text-align: center;
}

.formula {
  font-size: 1.1em;
  font-family: "MathJax_Main", "Times New Roman", serif;
  color: #d32f2f;
  line-height: 1.6;
}

.formula-explanation {
  font-size: 0.95em;
  line-height: 1.6;
}

.formula-explanation ul {
  padding-left: 20px;
}

.formula-explanation li {
  margin-bottom: 5px;
}


/* 精美目录样式 */
.toc-container {
  position: fixed;
  right: 2rem;
  top: 50%;
  transform: translateY(-50%);
  width: 240px;
  background: rgba(255, 255, 255, 0.98);
  border-radius: 12px;
  padding: 0;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
  backdrop-filter: blur(8px);
  border: 1px solid rgba(255, 255, 255, 0.2);
  z-index: 100;
  max-height: 80vh;
  overflow: hidden;
  transition: all 0.3s ease;
}

.toc-header {
  padding: 1rem 1.25rem;
  background: linear-gradient(135deg, #e5ecec 0%, #dae1e8 100%);
  color: white;
  position: relative;
}

.toc-title {
  display: flex;
  align-items: center;
  font-size: 1rem;
  font-weight: 600;
  margin: 0;
}

.toc-icon {
  width: 20px;
  height: 20px;
  margin-right: 0.5rem;
  fill: currentColor;
}

.toc-progress-bar {
  position: absolute;
  right: 0;
  top: 0;
  bottom: 0;
  width: 4px;
  background: rgba(255, 255, 255, 0.2);
}

.toc-progress {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  background: white;
  transition: height 0.1s linear;
}

.toc-list {
  list-style: none;
  padding: 0.75rem 0;
  margin: 0;
  max-height: calc(80vh - 60px);
  overflow-y: auto;
}

.toc-item {
  position: relative;
}

.toc-item-main {
  display: flex;
  align-items: center;
  padding: 0.5rem 1.25rem;
  cursor: pointer;
  transition: all 0.2s ease;
  color: #4a5568;
}

.toc-item:hover .toc-item-main {
  background: rgba(237, 242, 247, 0.5);
}

.toc-item.current > .toc-item-main,
.toc-item.child-current > .toc-item-main {
  color: #3182ce;
  font-weight: 500;
}

.toc-item.current > .toc-item-main {
  background: rgba(226, 232, 240, 0.7);
}

.toc-indicator {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #cbd5e0;
  margin-right: 0.75rem;
  transition: all 0.2s ease;
}

.toc-item.current > .toc-item-main > .toc-indicator,
.toc-item.child-current > .toc-item-main > .toc-indicator {
  background: #3182ce;
  transform: scale(1.3);
}

.toc-text {
  font-size: 0.9rem;
  flex-grow: 1;
}

.toc-arrow {
  margin-left: auto;
  transition: transform 0.2s ease;
  color: #a0aec0;
}

.toc-item.expanded > .toc-item-main > .toc-arrow {
  transform: rotate(180deg);
}

/* 二级目录样式 */
.toc-sublist {
  list-style: none;
  padding: 0;
  margin: 0;
  overflow: hidden;
  transition: max-height 0.3s ease;
  border-left: 2px solid #e2e8f0;
  margin-left: 1.5rem;
}

.toc-subitem {
  padding: 0.35rem 1.25rem 0.35rem 1.75rem;
  cursor: pointer;
  display: flex;
  align-items: center;
  transition: all 0.2s ease;
  color: #718096;
}

.toc-subitem:hover {
  background: rgba(237, 242, 247, 0.5);
  color: #4a5568;
}

.toc-subitem.current {
  background: rgba(226, 232, 240, 0.5);
  color: #3182ce;
  font-weight: 500;
}

.toc-subitem.current .toc-indicator {
  background: #3182ce;
  transform: scale(1.2);
}

/* 响应式调整 */
@media (max-width: 1200px) {
  .toc-container {
    right: 1rem;
    width: 220px;
  }
}

@media (max-width: 992px) {
  .toc-container {
    display: none;
  }
}
</style>

<style scoped>
.toc-container {
  position: fixed;
  right: 2rem;
  top: 50%;
  transform: translateY(-50%);
  width: 220px;
  background: rgba(255, 255, 255, 0.95);
  border-radius: 12px;
  padding: 1rem;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
  backdrop-filter: blur(5px);
  border: 1px solid rgba(0, 0, 0, 0.05);
  z-index: 100;
  transition: all 0.3s ease;
  max-height: 80vh;
  overflow-y: auto;
}

.toc-title {
  font-size: 1.1rem;
  font-weight: 600;
  color: #2d3748;
  padding-bottom: 0.5rem;
  margin-bottom: 0.5rem;
  border-bottom: 1px solid #e2e8f0;
}

.toc-list {
  list-style: none;
  padding: 0;
  margin: 0;
}

.toc-list > li {
  margin-bottom: 0.25rem;
}

.toc-item-main {
  padding: 0.5rem 0;
  cursor: pointer;
  display: flex;
  align-items: center;
  transition: all 0.2s ease;
  border-radius: 6px;
  padding-left: 0.5rem;
  position: relative;
}

.toc-list > li:hover > .toc-item-main {
  background: rgba(237, 242, 247, 0.7);
}

.toc-list > li.active > .toc-item-main {
  background: rgba(226, 232, 240, 0.7);
}

.toc-dot {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #cbd5e0;
  margin-right: 0.75rem;
  transition: all 0.2s ease;
}

.toc-list > li.active > .toc-item-main > .toc-dot {
  background: #4299e1;
  transform: scale(1.3);
}

.toc-text {
  font-size: 0.9rem;
  color: #4a5568;
  transition: all 0.2s ease;
  flex-grow: 1;
}

.toc-list > li.active > .toc-item-main > .toc-text {
  color: #2b6cb0;
  font-weight: 500;
}

.toc-arrow {
  margin-left: auto;
  transition: transform 0.2s ease;
  color: #a0aec0;
}

.toc-list > li.expanded > .toc-item-main > .toc-arrow {
  transform: rotate(180deg);
}

/* 二级目录样式 */
.toc-sublist {
  list-style: none;
  padding: 0;
  margin: 0 0 0 1.5rem;
  border-left: 1px dashed #e2e8f0;
}

.toc-sublist li {
  padding: 0.4rem 0;
  cursor: pointer;
  display: flex;
  align-items: center;
  transition: all 0.2s ease;
  border-radius: 6px;
  padding-left: 0.5rem;
}

.toc-sublist li:hover {
  background: rgba(237, 242, 247, 0.5);
}

.toc-sublist li.active {
  background: rgba(226, 232, 240, 0.5);
}

.toc-sublist li.active .toc-dot {
  background: #4299e1;
  transform: scale(1.2);
}

.toc-sublist li.active .toc-text {
  color: #2b6cb0;
  font-weight: 500;
}

/* 响应式调整 */
@media (max-width: 1200px) {
  .toc-container {
    right: 1rem;
    width: 200px;
  }
}

@media (max-width: 992px) {
  .toc-container {
    display: none;
  }
}


.resources-page {
  padding: 2rem 0;
  background-color: #f8fafc;
}

.resources-container {
  max-width: 1000px;
  margin: 0 auto;
  padding: 0 1.5rem;
}

.resource-section {
  margin-bottom: 3rem;
}

.resource-category {
  margin-bottom: 2rem;
  padding: 1.5rem;
  background: #ffffff;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
}

h2 {
  color: #2d3748;
  font-size: 1.8rem;
  margin-bottom: 1.5rem;
  padding-bottom: 0.5rem;
  border-bottom: 2px solid #e2e8f0;
}

h3 {
  color: #4a5568;
  font-size: 1.4rem;
  margin-bottom: 1rem;
}

h4 {
  color: #2d3748;
  margin-top: 0;
  font-size: 1.2rem;
}

.gan-intro {
  display: flex;
  gap: 2rem;
  margin-bottom: 2rem;
  align-items: flex-start;
  background: #f8fafc;
  padding: 1.5rem;
  border-radius: 8px;
}

.gan-text {
  flex: 1;
}

.gan-image {
  flex: 1;
  max-width: 500px;
  text-align: center;
}

.gan-architecture-img {
  max-width: 100%;
  height: auto;
  border-radius: 6px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.image-caption {
  margin-top: 0.5rem;
  font-size: 0.85rem;
  color: #64748b;
  text-align: center;
}

.resource-list {
  list-style: none;
  padding: 0;
}

.resource-item {
  margin-bottom: 1.5rem;
  padding: 1rem;
  transition: all 0.2s ease;
}

.resource-item:hover {
  background: #f7fafc;
  border-radius: 8px;
}

.resource-link {
  display: flex;
  align-items: center;
  color: #3182ce;
  font-weight: 500;
  text-decoration: none;
  font-size: 1.1rem;
  margin-bottom: 0.5rem;
}

.resource-link:hover {
  color: #2c5282;
  text-decoration: underline;
}

.resource-icon {
  margin-right: 0.5rem;
  font-size: 1.2rem;
}

.paper-meta {
  display: flex;
  gap: 1rem;
  margin: 0.5rem 0;
  font-size: 0.9rem;
  color: #64748b;
}

.resource-description {
  color: #4a5568;
  margin: 0.5rem 0 0 1.5rem;
  font-size: 0.95rem;
}

.resource-highlight {
  background: #f1f5f9;
  padding: 1rem;
  border-left: 3px solid #3b82f6;
  margin-top: 0.8rem;
  border-radius: 0 4px 4px 0;
}

.resource-highlight h5 {
  margin-top: 0;
  margin-bottom: 0.5rem;
  color: #1e40af;
}

.resource-highlight ul {
  margin: 0;
  padding-left: 1.2rem;
}

.resource-highlight li {
  margin-bottom: 0.3rem;
}

.code-demo {
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  margin-top: 1rem;
}

.code-tabs {
  display: flex;
  background: #f1f5f9;
  padding: 0.5rem;
}

.code-tabs button {
  padding: 0.5rem 1rem;
  border: none;
  background: none;
  cursor: pointer;
  font-weight: 500;
  color: #64748b;
  border-radius: 4px;
  margin-right: 0.5rem;
}

.code-tabs button.active {
  background: #ffffff;
  color: #3182ce;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.code-content {
  background: #2d3748;
  color: #f8fafc;
}

.code-block {
  padding: 1rem;
  overflow-x: auto;
}

pre {
  margin: 0;
  font-family: 'Fira Code', 'Courier New', monospace;
  font-size: 0.9rem;
  line-height: 1.5;
  white-space: pre-wrap;
}

@media (max-width: 768px) {
  .resources-container {
    padding: 0 1rem;
  }

  h2 {
    font-size: 1.5rem;
  }

  h3 {
    font-size: 1.2rem;
  }

  .gan-intro {
    flex-direction: column;
  }

  .gan-image {
    max-width: 100%;
    margin-top: 1rem;
  }
}


</style>
