use crate::Message;
use crate::image_decoder::{bgr_to_rgba, decode_named_images, grayscale_to_rgba};
use iced::futures;
use iced::futures::SinkExt;
use iced::futures::channel::mpsc;
use iced::widget::image;
use rs_ipc::SharedMessageMapper;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

#[derive(Debug)]
pub struct Frame {
    pub name: String,
    pub image: image::Handle,
}

struct RawImageData {
    name: String,
    width: u32,
    height: u32,
    pixels: Box<[u8]>,
}

pub fn image_producer(
    shared_memory: Arc<SharedMessageMapper>,
) -> impl futures::Stream<Item = Message> {
    let (mut sender, receiver) = mpsc::channel(1);

    thread::spawn(move || {
        shared_memory.add_reader();
        let mut last_read_version = 0;

        while !shared_memory.is_stopped() {
            let mut result = None;

            let mut start = Instant::now();
            shared_memory.blocking_read(last_read_version, |new_version, data| {
                last_read_version = new_version;

                start = Instant::now();
                let images: Vec<_> = decode_named_images(data)
                    .into_iter()
                    .map(|image| RawImageData {
                        name: image.name,
                        width: image.width,
                        height: image.height,
                        pixels: match image.channels {
                            1 => grayscale_to_rgba(image.pixels),
                            3 => bgr_to_rgba(image.pixels),
                            _ => panic!("Unknown image format with {} channels", image.channels),
                        },
                    })
                    .collect();


                result = Some(images);
            });

            let Some(result) = result else {
                continue;
            };
            println!("{:?}", start.elapsed());

            let frames: Vec<_> = result
                .into_iter()
                .map(|image| Frame {
                    name: image.name,
                    image: image::Handle::from_rgba(image.width, image.height, image.pixels),
                })
                .collect();

            if futures::executor::block_on(sender.send(Message::NewFrames(frames))).is_err() {
                break;
            }
        }

        let _ = futures::executor::block_on(sender.send(Message::VideoFinished));
    });

    receiver
}
