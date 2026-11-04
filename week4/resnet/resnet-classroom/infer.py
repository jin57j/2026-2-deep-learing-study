"""제공된 최종 가중치로 A부터 E까지 순서대로 확인한다."""

import _bootstrap  # week4 공통 코드 경로 설정

from a_base.infer import main as infer_base
from b_domain.infer import main as infer_domain
from c_augmentation.infer import main as infer_augmentation
from d_dog_breeds.infer import main as infer_dog_breeds
from e_localization.infer import main as infer_localization
from shared.plotting import plt


def main():
    infer_base()
    infer_domain()
    infer_augmentation()
    infer_dog_breeds()
    infer_localization()
    plt.show()


if __name__ == "__main__":
    main()
