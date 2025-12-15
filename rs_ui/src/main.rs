mod decoder;
mod grid;
mod image_reader;

use crate::grid::images_grid;
use crate::image_reader::{Frames, image_producer};
use iced::widget::{button, column, image, text};
use iced::{Alignment, Element, Length, Task};
use std::process::{Child, Command};

fn main() -> iced::Result {
    iced::application(AutoVision::new, AutoVision::update, AutoVision::view).title("AutoVision").run()
}

struct AutoVision {
    child: Child,
    frame: u64,
    fullscreen: bool,
    current_images: Frames,
}

impl Drop for AutoVision {
    fn drop(&mut self) {
        if let Err(e) = self.child.wait() {
            eprintln!("Error waiting Python process: {}", e);
        }
    }
}

#[derive(Debug, Clone)]
pub enum Message {
    NewFrames(Frames),
    ToggleFullscreen,
    VideoFinished,
}

impl AutoVision {
    fn new() -> (Self, Task<Message>) {
        let child = Command::new("python3")
            .arg("main_rust.py")
            .spawn()
            .expect("Failed to execute python3");

        let mut initial_state = Self {
            child,
            fullscreen: false,
            frame: 0,
            current_images: Frames::default(),
        };

        initial_state.current_images.description = String::from("Starting...");

        (initial_state, Task::stream(image_producer()))
    }

    fn update(&mut self, message: Message) {
        match message {
            Message::NewFrames(images) => {
                self.frame += 1;
                self.current_images = images;
            }
            Message::ToggleFullscreen => self.fullscreen = !self.fullscreen,
            Message::VideoFinished => self.frame = u64::MAX,
        }
    }

    fn view(&'_ self) -> Element<'_, Message> {
        let mut content = column![
            text(if self.frame == u64::MAX {
                "VIDEO FINISHED"
            } else {
                &self.current_images.description
            })
            .size(20)
            .font(iced::font::Font::MONOSPACE),
        ]
        .align_x(Alignment::Center)
        .width(Length::Fill)
        .height(Length::Fill);

        let frames = &self.current_images.frames;
        if !frames.is_empty() {
            let main_image_handle = self.current_images.frames[0].image.clone();
            let main_image = image(main_image_handle).height(Length::FillPortion(2));
            let main_image_button = button(main_image)
                .style(|_, _| button::Style::default())
                .on_press(Message::ToggleFullscreen);
            content = content.push(main_image_button);
        }

        if !self.fullscreen && frames.len() > 1 {
            content = content.push(images_grid(&self.current_images.frames[1..], 3));
        }

        content.into()
    }
}
