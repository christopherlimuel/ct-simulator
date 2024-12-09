import numpy as np
from skimage.transform import rotate
import cv2
from scipy.fft import fft, ifft
from scipy.interpolate import RectBivariateSpline

def process_image(img):
    img_height, img_width = img.shape[0], img.shape[1]
    size = max(img_height, img_width)
    square_img = np.full((size, size, 3), 0, dtype=np.uint8)
    
    # Tempelkan gambar asli di tengah kanvas
    y_offset = (size - img_height) // 2
    x_offset = (size - img_width) // 2
    square_img[y_offset:y_offset+img_height, x_offset:x_offset+img_width] = img
    # Ubah gambar menjadi greyscale
    grey_img = cv2.cvtColor(square_img, cv2.COLOR_BGR2GRAY)
    
    # Resize gambar ke ukuran yang diinginkan
    resized_img = cv2.resize(grey_img, (size, size), interpolation=cv2.INTER_AREA)
    return resized_img

def create_phantom(size, lp_cm, pixel_cm):
    """
    Membuat line pair phantom.
    size: ukuran gambar (pixel)
    lp_cm: resolusi line pairs per cm
    pixel_per_cm: resolusi pixel per cm
    """
    pattern_size = int(pixel_cm / lp_cm)  # Ukuran satu line pair (hitam + putih)
    phantom = np.zeros((size, size))

    for i in range(0, size, pattern_size * 2):
        phantom[:, i:i + pattern_size] = 1  # Bagian hitam
    return phantom

# =================================  THE OG   ==================================
# def generate_sinogram(image, num_detectors, num_rotation, kVp=1, mA=1):
#     dr = image.shape[0]/num_detectors
#     theta_step = 180/num_rotation

#     # Sumbu rotasi (array sudut dalam derajat)
#     thetas = np.arange(0, 180, theta_step)
    
#     # Rotasi gambar untuk setiap sudut
#     rotations = np.array([rotate(image, theta) for theta in thetas])
    
#     # Profil serapan sepanjang sumbu x untuk setiap sudut
#     profiles = np.array([rotation.sum(axis=0) * dr for rotation in rotations]).T
#     # Down-sampling sesuai jumlah detektor
#     sinogram = np.zeros((num_detectors, len(thetas)))
#     for i in range(len(thetas)):
#         sinogram[:, i] = np.interp(
#             np.linspace(0, profiles.shape[0] - 1, num_detectors),
#             np.arange(profiles.shape[0]),
#             profiles[:, i]
#         )
    
#     # Posisi detektor dalam koordinat linier
#     detector_positions = np.linspace(-image.shape[0] / 2, image.shape[0] / 2, num_detectors)
    
#     return thetas, detector_positions, sinogram

def generate_sinogram(image, num_detectors, num_rotation, kVp=1, mA=1):
    dr = image.shape[0] / num_detectors  # Resolusi detektor
    theta_step = 180 / num_rotation  # Langkah sudut rotasi

    # Sumbu rotasi (array sudut dalam derajat)
    thetas = np.arange(0, 180, theta_step)
    
    # Rotasi gambar untuk setiap sudut
    rotations = np.array([rotate(image, theta, resize=False) for theta in thetas])
    
    # Profil serapan sepanjang sumbu x untuk setiap sudut
    old_profiles = np.array([rotation.sum(axis=0) * dr for rotation in rotations])
    
    # Array untuk menyimpan profil yang sudah dinormalisasi
    profiles = np.zeros((num_detectors, len(thetas)))
    
    # Melakukan interpolasi pada setiap profil agar sesuai dengan jumlah detektor
    for theta_index, profile in enumerate(old_profiles):
        # Interpolasi profil ke ukuran yang sesuai dengan jumlah detektor
        profiles[:, theta_index] = np.interp(
            np.linspace(0, len(profile) - 1, num_detectors),
            np.arange(len(profile)),
            profile
        )
    
    # Normalisasi setiap profil dengan membaginya dengan nilai maksimum
    profiles = profiles / np.max(profiles)
    
    # Menghitung eksponensial dari profil
    profiles = profiles + np.log((kVp ** 2) * mA)
    
    # Down-sampling sesuai jumlah detektor (interpolasi)
    sinogram = np.zeros((num_detectors, len(thetas)))
    for i in range(len(thetas)):
        sinogram[:, i] = np.interp(
            np.linspace(0, profiles.shape[0] - 1, num_detectors),
            np.arange(profiles.shape[0]),
            profiles[:, i]
        )

    # Posisi detektor dalam koordinat linier
    detector_positions = np.linspace(-image.shape[0] / 2, image.shape[0] / 2, num_detectors)
    
    return thetas, detector_positions, sinogram

def FBP(sinogram, thetas_degree):
    # Konversi derajat ke radian
    thetas = []
    for theta in thetas_degree:
        thetas.append(theta*np.pi/180)
    #===== MEMBUAT KOORDINAT PENAMPANG =====#
    xmin, xmax = -1, 1                    # skala koordinat
    ymin, ymax = -1, 1
    xpoints, ypoints = 480, 480        # resolusi
    xypoints = np.zeros((xpoints, ypoints))

    # Konstruksi grid
    x = np.linspace(xmin, xmax, xpoints)
    y = np.linspace(ymin, ymax, ypoints)
    X, Y = np.meshgrid(x, y)
    dtheta = np.diff(thetas)[0]
    # ===== FILTERING =====
    # Melakukan FFT pada sinogram yang telah di-downsampling
    P = fft(sinogram, axis=0)

    # Menghitung frekuensi
    nu = np.fft.fftfreq(P.shape[0], d=np.diff(x)[0])

    # Membuat filter ramp (untuk FBP)
    ramp_filter = np.abs(nu)

    # Terapkan filter ramp pada data Fourier
    integrand = P.T * ramp_filter
    integrand = integrand.T

    # Lakukan inverse FFT setelah filtering
    p_p = np.real(ifft(integrand, axis=0))

    # ===== INTERPOLASI =====
    # Sesuaikan grid x dengan profil sinogram
    x_interpol = np.linspace(xmin, xmax, sinogram.shape[0])  # Menyesuaikan x dengan jumlah detektor

    # Pastikan dimensi data interpolasi cocok dengan grid
    p_p_interp = RectBivariateSpline(x_interpol, thetas, p_p)

    # # Fungsi rekonstruksi untuk FBP
    # def get_f(x, y):
    #     # Kalkulasi proyeksi untuk titik (x, y) menggunakan interpolasi
    #     return p_p_interp(x * np.cos(thetas) + y * np.sin(thetas), thetas, grid=False).sum() * dtheta

    # # Menerapkan rekonstruksi ke grid X, Y
    # f = np.vectorize(get_f)(X, Y)

    # Kalkulasi koordinat untuk semua titik di grid
    x_rot = X.flatten()[:, None] * np.cos(thetas) + Y.flatten()[:, None] * np.sin(thetas)

    # Interpolasi seluruh grid
    f_values = p_p_interp(x_rot, thetas, grid=False)

    # Integrasi nilai-nilai dalam domain rotasi
    f = f_values.sum(axis=1).reshape(X.shape) * dtheta
    return f