## Carried over:

Session 15 addressed round 6: fixed the profile tier routing bug by adding force_tier kwarg to Router.decide() and decide_batch(). When set, overrides the active profile's model_preference for that call, taking precedence over request.force_tier. ov_server can drop the decide→reselect workaround (c5711b5). First release under strict semver: v0.2.0. 131/131 tests, published to PyPI, pushed to GitHub.
