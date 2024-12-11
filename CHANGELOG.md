# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2024-12-11
### Added
- `COSINE_SCORE_THRESHOLD` in `recommender.py` for to exclude poor matches (I suggest `0.6` is a reasonable starting value)
### Changed
- `recommender.py` uses Gemini 1.5 Flash, but `main.py` still on Gemini 1.0 Pro
- Updated README
- Updated `requirements.txt`

## [0.3.0] - 2024-05-02
### Added
- Retry logic when single outfit descriptions were generated rather than separate for each
### Changed
- Updated README
- Updated `requirements.txt`

## [0.2.0] - 2024-04-26
### Added
- Function `any_list_element_in_string(list, str)` to filter out responses involve multiple articles of clothing
- `main.py` which runs a (local) templated Flask frontend

## [0.1.0] - 2024-04-20
### Added
- Initial commit (giving this project its own repo!)
