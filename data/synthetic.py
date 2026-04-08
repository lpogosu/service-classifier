"""Генератор синтетического датасета объявлений.

Исходный набор кейса не распространяется, поэтому репозиторий поставляется
с генератором: он собирает объявления из фрагментов реальных формулировок
и знает разметку по построению — какие услуги в тексте и продаются ли они
отдельно. Набор нужен, чтобы пайплайн, обучение и метрики запускались
end-to-end; на цифры из README он не влияет.

    python -m data.synthetic --rows 400 --seed 20260412
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from service_classifier.config import DEFAULT_DATASET_PATH, SOURCE_MC_ID
from service_classifier.datasets import LabeledAd, write_dataset

DEFAULT_ROWS = 400
DEFAULT_SEED = 20260412
#: Доля объявлений, уходящих в отложенную часть.
TEST_FRACTION = 0.2
FIRST_ITEM_ID = 100_001

SOURCE_MC_TITLE = "Ремонт квартир и домов под ключ"

#: Формулировки услуг: мастер прямо предлагает работу.
SERVICE_PHRASES: dict[int, list[str]] = {
    102: [
        "монтаж и замена труб водоснабжения",
        "разводка сантехники и установка приборов",
        "сантехнические работы: стояки, канализация, отопление",
        "установка унитаза, ванны, смесителя",
    ],
    103: [
        "электромонтаж: щиток, розетки, освещение",
        "полная замена электропроводки",
        "электрика от штробления до подключения автоматов",
        "монтаж электрики с составлением схемы",
    ],
    104: [
        "натяжные потолки с замером в день обращения",
        "монтаж натяжных потолков ПВХ",
        "натяжной потолок в комнату или санузел",
    ],
    105: [
        "укладка плитки и керамогранита",
        "плиточные работы: стены, полы, фартук",
        "облицовка плиткой с затиркой швов",
        "укладка кафеля в ванной",
    ],
    106: [
        "поклейка обоев любых типов",
        "оклейка обоев с подготовкой стен",
        "поклейка флизелиновых обоев",
    ],
    107: [
        "малярные работы и покраска стен",
        "покраска потолков и дверей",
        "декоративная штукатурка и окраска",
    ],
    108: [
        "штукатурка стен по маякам",
        "механизированная штукатурка и шпаклевка",
        "выравнивание стен под покраску",
    ],
    109: [
        "укладка ламината и линолеума",
        "стяжка пола и напольные покрытия",
        "укладка кварцвинила с подложкой",
    ],
    110: [
        "монтаж гипсокартона: перегородки, короба, ниши",
        "работы с гипсокартоном и профилем",
        "подвесной потолок из ГКЛ",
    ],
    111: [
        "демонтажные работы любой сложности",
        "снос перегородок и демонтаж стен",
        "демонтаж отделки с выносом мусора",
    ],
}

#: Те же категории, но как объекты сноса — услуги здесь нет. Такие тексты
#: и ловит контекстная верификация стадии 1.5.
DEMOLITION_OBJECTS: dict[int, list[str]] = {
    105: ["сбиваем плитку и кафель", "демонтаж плитки со стен и пола"],
    106: ["снимаем старые обои", "удаление обоев со стен"],
    108: ["сбивка штукатурки до основания", "демонтаж штукатурки"],
    109: ["срываем линолеум и ламинат", "демонтаж старой стяжки"],
    110: ["разбираем конструкции из гипсокартона", "демонтаж коробов ГКЛ"],
}

SPECIALIST_OPENINGS = [
    "Мастер с собственным инструментом.",
    "Работаю сам, без посредников.",
    "Принимаю заказы по отдельным видам работ.",
    "Частный мастер, выезд по городу и области.",
]

SPECIALIST_CLOSINGS = [
    "Каждый вид работ считаю отдельно, смета прозрачная.",
    "Цена за квадратный метр — от 450 руб.",
    "Берусь и за отдельную услугу, и за несколько сразу.",
    "Работаю отдельно по каждому направлению, звоните.",
]

COMPREHENSIVE_OPENINGS = [
    "Выполняем ремонт квартир под ключ.",
    "Наша бригада делает комплексный ремонт от и до.",
    "Ремонт под ключ: берём объект целиком.",
    "Капитальный ремонт квартир и домов под ключ.",
]

COMPREHENSIVE_CLOSINGS = [
    "Все работы входят в общую смету, отдельно не считаем.",
    "Сдаём объект под чистовую отделку, гарантия по договору.",
    "Работаем по договору, включая все виды отделочных работ.",
    "Полный цикл: от демонтажа до финишной отделки.",
]

DEMOLITION_OPENINGS = [
    "Демонтаж квартир и домов, работаем бригадой.",
    "Занимаемся только демонтажом и сносом.",
    "Демонтажные работы: подготовка помещения к ремонту.",
]

DEMOLITION_CLOSINGS = [
    "Вывоз мусора и уборка после работ включены.",
    "Алмазная резка проёмов, слом стен, разборка конструкций.",
    "Работаем ночью и в выходные, есть допуск.",
]

LIST_MARKERS = ["- ", "• ", "✔ ", ""]

#: Латиница, визуально неотличимая от кириллицы: воспроизводим подмену,
#: встречавшуюся в исходных данных.
HOMOGLYPHS = {"а": "a", "е": "e", "о": "o", "с": "c", "р": "p", "к": "k", "х": "x"}


def _typo(word: str, rng: random.Random) -> str:
    """Переставляет две соседние буквы — самая частая опечатка в наборе."""
    if len(word) < 4:
        return word
    index = rng.randrange(1, len(word) - 2)
    chars = list(word)
    chars[index], chars[index + 1] = chars[index + 1], chars[index]
    return "".join(chars)


def _homoglyph(text: str, rng: random.Random) -> str:
    cyrillic, latin = rng.choice(list(HOMOGLYPHS.items()))
    return text.replace(cyrillic, latin, 1)


def _apply_noise(text: str, rng: random.Random) -> str:
    """Опечатки, гомоглифы и эмодзи — то, чем реальные объявления отличаются от чистых."""
    if rng.random() < 0.15:
        words = text.split()
        index = rng.randrange(len(words))
        words[index] = _typo(words[index], rng)
        text = " ".join(words)
    if rng.random() < 0.16:
        text = _homoglyph(text, rng)
    if rng.random() < 0.20:
        text = f"{text}\n{rng.choice(['🔨', '⚡', '🏠'])} Звоните в любое время."
    return text


def _bullet_block(phrases: list[str], rng: random.Random) -> str:
    marker = rng.choice(LIST_MARKERS)
    if marker:
        return "\n".join(f"{marker}{phrase}" for phrase in phrases)
    return ", ".join(phrases) + "."


def _specialist_ad(rng: random.Random) -> tuple[str, list[int], bool, str]:
    mc_ids = rng.sample(sorted(SERVICE_PHRASES), rng.randint(2, 4))
    phrases = [rng.choice(SERVICE_PHRASES[mc_id]) for mc_id in mc_ids]
    text = "\n".join(
        [
            rng.choice(SPECIALIST_OPENINGS),
            _bullet_block(phrases, rng),
            rng.choice(SPECIALIST_CLOSINGS),
        ]
    )
    return text, sorted(mc_ids), True, "specialist"


def _comprehensive_ad(rng: random.Random) -> tuple[str, list[int], bool, str]:
    mc_ids = rng.sample(sorted(SERVICE_PHRASES), rng.randint(5, 8))
    phrases = [rng.choice(SERVICE_PHRASES[mc_id]) for mc_id in mc_ids]
    text = "\n".join(
        [
            rng.choice(COMPREHENSIVE_OPENINGS),
            "В комплексный ремонт включая все этапы входит:",
            _bullet_block(phrases, rng),
            rng.choice(COMPREHENSIVE_CLOSINGS),
        ]
    )
    return text, sorted(mc_ids), False, "comprehensive"


def _demolition_ad(rng: random.Random) -> tuple[str, list[int], bool, str]:
    """Чисто демонтажное объявление: другие категории упомянуты как объекты сноса."""
    objects = rng.sample(sorted(DEMOLITION_OBJECTS), rng.randint(2, 4))
    phrases = [rng.choice(DEMOLITION_OBJECTS[mc_id]) for mc_id in objects]
    text = "\n".join(
        [
            rng.choice(DEMOLITION_OPENINGS),
            "ДЕМОНТАЖ, СЛОМ, СНОС:",
            _bullet_block(phrases, rng),
            "Демонтаж стен, демонтаж перегородок, демонтаж потолка, демонтаж полов.",
            rng.choice(DEMOLITION_CLOSINGS),
        ]
    )
    return text, [111], False, "demolition"


def _single_service_ad(rng: random.Random) -> tuple[str, list[int], bool, str]:
    mc_id = rng.choice(sorted(SERVICE_PHRASES))
    text = "\n".join(
        [
            rng.choice(SPECIALIST_OPENINGS),
            rng.choice(SERVICE_PHRASES[mc_id]).capitalize() + ".",
            rng.choice(SPECIALIST_CLOSINGS),
        ]
    )
    return text, [mc_id], True, "single"


#: Доли архетипов подобраны так, чтобы shouldSplit=true встречался примерно
#: в трети объявлений — как в исходном наборе.
ARCHETYPES = (
    (_specialist_ad, 0.28),
    (_single_service_ad, 0.10),
    (_comprehensive_ad, 0.47),
    (_demolition_ad, 0.15),
)


def generate(rows: int = DEFAULT_ROWS, seed: int = DEFAULT_SEED) -> list[LabeledAd]:
    """Собирает воспроизводимый набор размеченных объявлений."""
    rng = random.Random(seed)
    builders = [builder for builder, _ in ARCHETYPES]
    weights = [weight for _, weight in ARCHETYPES]

    ads: list[LabeledAd] = []
    for offset in range(rows):
        builder = rng.choices(builders, weights=weights, k=1)[0]
        text, mc_ids, should_split, case_type = builder(rng)
        text = _apply_noise(text, rng)
        ads.append(
            LabeledAd(
                item_id=FIRST_ITEM_ID + offset,
                source_mc_id=SOURCE_MC_ID,
                source_mc_title=SOURCE_MC_TITLE,
                description=text,
                target_detected_mc_ids=mc_ids,
                target_split_mc_ids=mc_ids if should_split else [],
                should_split=should_split,
                case_type=case_type,
                split="test" if rng.random() < TEST_FRACTION else "train",
            )
        )
    return ads


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m data.synthetic",
        description="Генерация синтетического датасета объявлений",
    )
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_DATASET_PATH)
    args = parser.parse_args()

    ads = generate(rows=args.rows, seed=args.seed)
    write_dataset(args.output, ads)

    positives = sum(1 for ad in ads if ad.should_split)
    print(
        f"Сгенерировано {len(ads)} объявлений -> {args.output}\n"
        f"  shouldSplit=true: {positives} ({positives / len(ads):.0%})\n"
        f"  seed={args.seed}"
    )


if __name__ == "__main__":
    main()
