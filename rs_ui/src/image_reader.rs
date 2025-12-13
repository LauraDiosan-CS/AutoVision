use crate::Message;
use crate::decoder::{bgr_to_rgba, decode_named_images, grayscale_to_rgba};
use iced::futures;
use iced::futures::SinkExt;
use iced::futures::channel::mpsc;
use iced::widget::image;
use rs_ipc::SharedMessageMapper;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

#[derive(Debug, Clone, Default)]
pub struct Frames {
    pub description: String,
    pub frames: Vec<Frame>,
}

#[derive(Debug, Clone)]
pub struct Frame {
    pub name: String,
    pub image: image::Handle,
}

pub fn image_producer(
    shared_memory: Arc<SharedMessageMapper>,
) -> impl futures::Stream<Item = Message> {
    let (mut sender, receiver) = mpsc::channel(1);

    thread::spawn(move || {
        shared_memory.add_reader();

        let mut cache = BTreeMap::new();
        let mut last_read_version = 0;

        while !shared_memory.is_stopped() {
            let mut result = None;

            let mut start = Instant::now();
            shared_memory.blocking_read(last_read_version, |new_version, data| {
                last_read_version = new_version;

                start = Instant::now();
                let decoded = decode_named_images(data);
                let frames: Vec<_> = decoded
                    .images
                    .into_iter()
                    .map(|image| Frame {
                        name: image.name.to_string(),
                        image: image::Handle::from_rgba(
                            image.width,
                            image.height,
                            match image.channels {
                                1 => grayscale_to_rgba(image.pixels),
                                3 => bgr_to_rgba(image.pixels),
                                _ => {
                                    panic!("Unknown image format with {} channels", image.channels)
                                }
                            },
                        ),
                    })
                    .collect();

                result = Some(Frames {
                    description: decoded.description.to_string(),
                    frames,
                });
            });

            let Some(result) = result else {
                continue;
            };
            println!("{:?}", start.elapsed());

            // Cache the items
            for frame in result.frames {
                cache.insert(frame.name, frame.image);
            }

            let main_frame_name = "Main";
            let mut frames: Vec<_> = cache
                .iter()
                .map(|(name, image)| Frame {
                    name: name.clone(),
                    image: image.clone(),
                })
                .collect();
            if let Some(index) = frames
                .iter()
                .position(|frame| frame.name == main_frame_name)
            {
                let element = frames.remove(index);
                frames.insert(0, element);
            }
            let frames = Frames {
                description: result.description,
                frames,
            };

            if futures::executor::block_on(sender.send(Message::NewFrames(frames))).is_err() {
                break;
            }
        }

        let _ = futures::executor::block_on(sender.send(Message::VideoFinished));
    });

    receiver
}
