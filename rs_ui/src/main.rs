mod grid;
mod decoder;
mod image_reader;

use crate::grid::images_grid;
use crate::image_reader::{image_producer, Frames};
use iced::widget::{column, container, text};
use iced::{Element, Length, Task};
use rs_ipc::SharedMessageMapper;
use std::process::{Child, Command};
use std::sync::Arc;

fn main() -> iced::Result {
    iced::application("AutoVision", Counter::update, Counter::view).run_with(Counter::new)
}

struct Counter {
    child: Child,
    frame: u64,
    current_images: Frames,
    shared_memory: Arc<SharedMessageMapper>,
}

impl Drop for Counter {
    fn drop(&mut self) {
        self.shared_memory.stop();
        if let Err(e) = self.child.wait() {
            eprintln!("Error waiting Python process: {}", e);
        }
    }
}

#[derive(Debug)]
pub enum Message {
    NewFrames(Frames),
    VideoFinished,
}

impl Counter {
    fn new() -> (Self, Task<Message>) {
        let shared_memory = Arc::new(
            SharedMessageMapper::create(c"rust_ui".into(), 64 * 1024 * 1024)
                .expect("Can't create shared memory"),
        );

        let child = Command::new("python3")
            .arg("main_rust.py")
            // .stdout(Stdio::piped())
            .spawn()
            .expect("Failed to execute python3");

        let initial_state = Self {
            child,
            frame: 0,
            current_images: Frames::default(),
            shared_memory: shared_memory.clone(),
        };

        (initial_state, Task::stream(image_producer(shared_memory)))
    }

    fn update(&mut self, message: Message) {
        match message {
            Message::NewFrames(images) => {
                self.frame += 1;
                self.current_images = images;
            }
            Message::VideoFinished => {
                self.frame = u64::MAX;
            }
        }
    }

    fn view(&'_ self) -> Element<'_, Message> {
        let content = column![
            text(if self.frame == u64::MAX {
                "VIDEO FINISHED"
            } else {
                &self.current_images.description
            })
            .size(20)
            .font(iced::font::Font::MONOSPACE),
            images_grid(&self.current_images.frames, 3)
        ];

        container(content)
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
    }
}
