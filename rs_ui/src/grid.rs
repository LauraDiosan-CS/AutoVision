use crate::Message;
use crate::image_reader::Frame;
use iced::widget::container::background;
use iced::widget::space::horizontal;
use iced::widget::{column, container, image, row, text};
use iced::{Alignment, Background, Color, Element, Length, color};

const BACKGROUND_COLORS: [Color; 6] = [
    color!(250, 206, 104),
    color!(90, 156, 181),
    color!(250, 104, 104),
    color!(250, 172, 104),
    color!(255, 205, 201),
    color!(119, 136, 115),
];

pub fn images_grid(frames: &'_ [Frame], column_count: usize) -> Element<'_, Message> {
    column(
        frames
            .chunks(column_count)
            .enumerate()
            .map(|(chunk_index, chunk)| {
                let mut r = row(chunk.iter().enumerate().map(|(row_index, frame)| {
                    let index = (chunk_index * column_count) + row_index;
                    container(column![
                        text(&frame.name)
                            .width(Length::Fill)
                            .align_x(Alignment::Center),
                        image(frame.image.clone())
                            .width(Length::Fill)
                            .content_fit(iced::ContentFit::Contain),
                    ])
                    .style(move |_| {
                        background(Background::Color(
                            BACKGROUND_COLORS[index % BACKGROUND_COLORS.len()],
                        ))
                    })
                    .align_x(Alignment::Center)
                    .padding(2f32)
                    .into()
                }));

                for _ in 0..column_count - chunk.len() {
                    r = r.push(horizontal());
                }

                r.into()
            }),
    )
    .into()
}
