# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Setup
#
# First thing, read the instructions in the README to set up uv. You can, if you want, use Conda but uv is just so much faster and nicer tbh. I've added Conda instructions too but I really recommend just using uv. I've also added some instructions about using VSCode. Here it doesn't really matter, use your favorite editor (VSCode, PyCharm, straight up emacs or vim) but it's worth picking an editor instead of just doing it as one-off files. A good opportunity to learn a skill!

# %% [markdown] vscode={"languageId": "plaintext"}
# ## Problem 1 - 1-D Linear Regression warmup
# In the lectures, we've shown that there is an explicit solution to linear regression. We're going to play with making sure one can implement that, as well as the gradient descent version of it, and do some comparisons between them.

# %% [markdown]
#
