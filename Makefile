
install:
	pip install --update pip
	pip install "jax[tpu]>=0.3" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
	pip install -r requirements.txt
