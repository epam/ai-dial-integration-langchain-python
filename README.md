# Overview

`langchain_openai` [doesn't allow](https://github.com/langchain-ai/langchain/issues/26617) to pass extra request/response parameters to/from the upstream model.

The repo provides ways to overcome this issue.

## Minimal example

Find the minimal example highlighting the issue with `langchain_openai` at the [example folder](./example/):

```sh
> cd example
> python -m venv .venv
> source .venv/bin/activate
> pip install -q -r requirements.txt
> python -m app
Received extra fields in:
(1) ☐ Request - in the `messages` list
(2) ☑ Request - on the top-level
(3) ☐ Response - in the `message` field
(4) ☐ Response - on the top-level
```

`langchain_openai` ignores certain extra fields, meaning that the upstream endpoint won't receive (1) and the client won't receive (3) and (4) if they were sent by the upstream.

Note that **top-level request extra fields** do actually reach the upstream.

## Solution #1 *(monkey-patching the library)*

One way to *fix* the issue, is to modify the methods which ignore these extra fields and make the methods actually take them into account.

This is achieved via monkey-patching certain private methods in `langchain_openai` which do the conversion from the Langchain datatypes to dictionaries and vice versa.

### Usage

Copy [the patch modules](./src/aidial_integration_langchain/patch/) to your project, then import before any Langchain module is imported.

```sh
> cd example
> python -m venv .venv
> source .venv/bin/activate
> pip install -q -r requirements.txt
> cp -r ../src/aidial_integration_langchain/patch .
> python -m app
Received extra fields in:
(1) ☑ Request - in the `messages` list
(2) ☑ Request - on the top-level
(3) ☑ Response - in the `message` field
(4) ☑ Response - on the top-level
```

### Supported versions

The following `langchain_openai` versions have been tested for Python 3.9, 3.10, 3.11, 3.12, and 3.13:

|Version|Request per-message|Response per-message|Response top-level|
|---|---|---|---|
|>=0.1.1,<=0.1.22|🟢|🟢|🔴|
|>=0.1.23,<=0.1.25|🟢|🟢|🟢|
|>=0.2.0,<=0.2.14|🟢|🟢|🟢|
|>=0.3.0,<=0.3.16|🟢|🟢|🟢|

Note that `langchain_openai<=0.1.22` doesn't support response top-level extra fields, since the structure of the code back then was not very amicable for monkey-patching in this particular respect.

## Solution #2 *(custom AzureChatOpenAI class)*

The implementation of the `AzureChatOpenAI` class may be copied and modified as needed to take into account extra fields.

Find the redefined classes at [aidial_integration_langchain.langchain_openai](./src/aidial_integration_langchain/langchain_openai/).

### Usage

Simply import the `AzureChatOpenAI` class from this repo instead of `langchain_openai`:

```diff
# ./example/app.py
- from langchain_openai import AzureChatOpenAI
+ from aidial_integration_langchain.langchain_openai import AzureChatOpenAI
```

### Supported versions

Currently only `langchain_openai==0.2.0` is supported for Python 3.9, 3.10, 3.11 and 3.12.

## Environment variables

The list of extra fields that are allowed to pass-through is controlled by the following environment variables.

|Name|Default|
|---|---|
|LC_EXTRA_REQUEST_MESSAGE_FIELDS|custom_content|
|LC_EXTRA_RESPONSE_MESSAGE_FIELDS|custom_content|
|LC_EXTRA_RESPONSE_FIELDS|statistics|

Each contains a comma-separated list of field names.
