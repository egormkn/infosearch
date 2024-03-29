Соревнование по исправлению слов

Поисковой системе важно уметь исправлять запросы с ошибками. В этом соревновании предлагается решить немного упрощенную версию этой задачи: научится правильно исправлять отдельные слова.
Условия

Максимальное количество баллов за задачу — 25.

- Решения хуже «No Fixes At All» получают 0 баллов.
- Не более шести посылок в сутки.
- Чужие решения сдавать нельзя.
- Дедлайн указан на странице соревнования. После него сдавать ничего нельзя.

Частичные решения

- Префиксное дерево и эвристики с лекции — 10 баллов.
- Фонетические алгоритмы — 3 балла.
- Модель языка — 2 балла.
- Модель ошибок — 5 баллов (все уровни).
- Клёвые эвристики — 5 баллов (не менее двух).

Чтобы получить заслуженные баллы нужно:

- Выбить score лучше, чем «No Fixes At All» на private части данных.
- Послать код решения и короткое описание того, что именно вы сделали (какую часть выполнили и на сколько баллов рассчитываете).
- Если вы хотите получить дополнительные баллы за «клёвые эвристики», нужно дополнительно описать почему вы считаете их такими.

Файлы:

- no_fix.submission.csv — пример корректо сформированного файла посылки. В этой посылке ни одно слово не исправлено. Ожидается, что метрика качества любого решения будет выше, чем метрика качества этой посылки.
- public.freq.csv — файл с размеченными данными. Id — оригинальное слово, Expected — ожидаемое исправление слова (в случае, если исправление не требуется, значение сопадает с Id, Frequency — частота, с которой встретилось слово за X дней в логах Поиска.
- words.freq.csv — файл с частотами всех слов (размеченных и не размеченных).
