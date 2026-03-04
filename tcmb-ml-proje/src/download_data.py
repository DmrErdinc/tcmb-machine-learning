"""
Kaggle veri setini indirir ve data/raw klasörüne yerleştirir.
Eğer veri zaten mevcutsa indirme atlanır.
Kaggle CLI veya token yoksa anlaşılır hata mesajı verir.
"""

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from src.utils import KAGGLE_SLUG, VERI_HAM, VERI_KAYNAK, ensure_dirs, setup_logging

logger = setup_logging("download_data")


def kaggle_cli_mevcut() -> bool:
    """Kaggle CLI kurulu mu kontrol eder."""
    try:
        subprocess.run(["kaggle", "--version"],
                       capture_output=True, text=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def kaggle_token_mevcut() -> bool:
    """~/.kaggle/kaggle.json var mı kontrol eder."""
    token_yolu = Path.home() / ".kaggle" / "kaggle.json"
    return token_yolu.exists()


def yerel_veri_kopyala() -> bool:
    """
    Proje dışındaki dataset/ klasöründen data/raw'a kopyalar.
    Eğer dataset/ klasörü mevcutsa kullanır.
    """
    if not VERI_KAYNAK.exists():
        return False

    csv_dosyalar = list(VERI_KAYNAK.glob("*.csv"))
    if not csv_dosyalar:
        return False

    logger.info(f"Yerel veri kaynağı bulundu: {VERI_KAYNAK}")
    for csv in csv_dosyalar:
        hedef = VERI_HAM / csv.name
        if not hedef.exists():
            shutil.copy2(csv, hedef)
            logger.info(f"  Kopyalandı: {csv.name}")
        else:
            logger.info(f"  Zaten mevcut: {csv.name}")
    return True


def kaggle_indir() -> None:
    """Kaggle CLI ile veri setini indirir ve açar."""
    if not kaggle_cli_mevcut():
        logger.error(
            "Kaggle CLI bulunamadı!\n"
            "Kurulum: pip install kaggle\n"
            "Ayrıntılar için README.md dosyasına bakınız."
        )
        sys.exit(1)

    if not kaggle_token_mevcut():
        logger.error(
            "Kaggle API token bulunamadı!\n"
            "1) https://www.kaggle.com/settings adresine gidin\n"
            "2) 'Create New Token' butonuna tıklayın\n"
            "3) İndirilen kaggle.json dosyasını ~/.kaggle/ altına kopyalayın\n"
            "Ayrıntılar için README.md dosyasına bakınız."
        )
        sys.exit(1)

    logger.info(f"Kaggle'dan indiriliyor: {KAGGLE_SLUG}")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_SLUG,
         "-p", str(VERI_HAM), "--unzip"],
        check=True,
    )
    logger.info("İndirme tamamlandı.")


def main() -> None:
    parser = argparse.ArgumentParser(description="TCMB veri seti indirme aracı")
    parser.add_argument("--force", action="store_true",
                        help="Mevcut veriyi yeniden indir")
    args = parser.parse_args()

    ensure_dirs()

    mevcut_csv = list(VERI_HAM.glob("*.csv"))
    if mevcut_csv and not args.force:
        logger.info(f"Veri zaten mevcut ({len(mevcut_csv)} dosya). "
                    f"Yeniden indirmek için --force kullanın.")
        return

    # Önce yerel kaynaktan kopyalamayı dene
    if yerel_veri_kopyala():
        logger.info("Veri yerel kaynaktan başarıyla kopyalandı.")
        return

    # Yerel kaynak yoksa Kaggle'dan indir
    kaggle_indir()


if __name__ == "__main__":
    main()
