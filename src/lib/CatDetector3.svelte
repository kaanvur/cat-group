<script lang="ts">
	import * as tf from '@tensorflow/tfjs';
	import * as mobilenet from '@tensorflow-models/mobilenet';
	import { onMount } from 'svelte';

	let image1: HTMLImageElement;
	let image2: HTMLImageElement;
	let similarityScore: number | null = null;
	let model: mobilenet.MobileNet;
	let image1Loaded = false;
	let image2Loaded = false;

	function handleImageUpload(
		event: Event,
		imageElement: HTMLImageElement,
		setImageLoaded: (value: boolean) => void
	) {
		const file = (event.target as HTMLInputElement).files[0];
		if (file) {
			const reader = new FileReader();
			reader.onload = (e) => {
				imageElement.src = e.target.result as string;
				imageElement.onload = () => setImageLoaded(true);
			};
			reader.readAsDataURL(file);
		}
	}

	async function compareImages() {
		if (!image1Loaded || !image2Loaded) {
			console.error('Both images must be fully loaded.');
			return;
		}

		const img1 = tf.browser.fromPixels(image1).toFloat();
		const img2 = tf.browser.fromPixels(image2).toFloat();

		const features1 = model.infer(img1, true).flatten();
		const features2 = model.infer(img2, true).flatten();

		const similarity = features1.dot(features2).div(features1.norm().mul(features2.norm()));
		similarityScore = similarity.dataSync()[0];
	}

	onMount(async () => {
		model = await mobilenet.load();
		console.log('MobileNet model loaded');
	});

	$: if (image1Loaded && image2Loaded) {
		compareImages();
	}
</script>

<main>
	<input
		type="file"
		accept="image/*"
		on:change={(event) => handleImageUpload(event, image1, (value) => (image1Loaded = value))}
	/>
	<input
		type="file"
		accept="image/*"
		on:change={(event) => handleImageUpload(event, image2, (value) => (image2Loaded = value))}
	/>
	<div>
		<img bind:this={image1} alt="Cat 1" />
		<img bind:this={image2} alt="Cat 2" />
	</div>
	{#if similarityScore !== null}
		<p>Similarity Score: {similarityScore}</p>
	{/if}
</main>

<style>
	img {
		width: 200px;
		height: auto;
		margin: 10px;
	}
</style>
