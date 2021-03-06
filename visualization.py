import matplotlib.pyplot as plt
import matplotlib.image as mmg

original = mmg.imread('leverkusen_000057_000019_leftImg8bit_original.png')
predicted = mmg.imread('leverkusen_000057_000019_leftImg8bit.png')
original1 = mmg.imread('leverkusen_000056_000019_leftImg8bit.png')
predicted1 = mmg.imread('leverkusen_000056_000019_leftImg8bit_test.png')
plt.figure()
plt.subplots_adjust(wspace =0, hspace =0)
plt.subplot(2, 2, 1)
plt.imshow(original)
plt.axis(False)
plt.subplot(2, 2, 2)
plt.imshow(predicted)
plt.axis(False)
plt.subplot(2, 2, 3)
plt.imshow(original1)
plt.axis(False)
plt.subplot(2, 2, 4)
plt.imshow(predicted1)
plt.axis(False)
plt.show()